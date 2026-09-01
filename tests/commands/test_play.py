"""Tests for `-play` (src/commands/play.py)."""

import asyncio
import contextlib
from collections.abc import AsyncIterator, Coroutine
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
import redis.asyncio as aioredis

from src import play_pipeline
from src.guild_state import Analytics
from src.musicbot import MusicBot
from src.commands import play as play_cmd
from src.recovery import abandon_cold_start
from src.guild_queue import RemoveMode, RemoveOutcome
from src.play_placement import (
    PLACE_TIMEOUT_SECS,
    PlaceResult,
    ResolveMode,
    Placement,
    PlayMode,
    PlayRequest,
    _GuildPlays,
    play_key,
    resolve_mode_for,
)
from src.musicplayer import (
    RESTORE_WAIT_SECS,
    _START_WRITE_TIMEOUT,
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
    admit,
    command_callback,
    connected_vc,
    in_authors_channel,
    recording_span,
    settle,
    song,
    mock_mp,
    no_typing,
    paused_vc,
    playing_vc,
    queue_object,
)

# Captured before any test replaces the module attribute: the tests that want the
# REAL insert (the depth is minted there) restore it by name, and the seam every
# other test stubs is `play_pipeline.enqueue_single`.
_REAL_ENQUEUE_SINGLE = play_pipeline.enqueue_single


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
            mock_ctx.voice_client = connected_vc(mock_ctx)  # what a real join leaves
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
        mock_ctx.voice_client = playing_vc(mock_ctx)
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

    async def test_the_warm_path_hands_queue_source_the_placement_mode(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The mode has no default, but it could still be hardcoded at one call site
        and derived at the other. Both must read it from the placement."""
        mock_ctx.voice_client = playing_vc(mock_ctx)
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task"),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert play_pipeline.queue_source.await_args is not None
        # The literal, not resolve_mode_for(TAIL): comparing production against the
        # same function passes however the call site reads its placement.
        assert (
            play_pipeline.queue_source.await_args.kwargs["mode"] is ResolveMode.FLAT_OK
        )
        assert resolve_mode_for(Placement.TAIL) is ResolveMode.FLAT_OK

    async def test_the_cold_path_hands_queue_source_the_placement_mode(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        fake_qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        play_pipeline.queue_source = AsyncMock(return_value=fake_qobj)
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = connected_vc(mock_ctx)
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert play_pipeline.queue_source.await_args is not None
        assert play_pipeline.queue_source.await_args.kwargs["mode"] is ResolveMode.FULL
        assert resolve_mode_for(Placement.COLD_FRONT) is ResolveMode.FULL


class TestPlayAnalytics:
    """The ask-time Analytics -play mints and hands to queue_source.

    Asserted on the call rather than on a returned object: queue_source is what
    carries the value into every construction site, and nothing downstream
    restamps it, so the hand-off IS the behavior."""

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
            mock_ctx.voice_client = connected_vc(mock_ctx)
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
        mock_ctx.voice_client = playing_vc(mock_ctx)
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

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        queued = mp.queue_put.await_args.args[0]
        assert queued.analytics.queue_position == 12

    async def test_warm_path_carries_the_ask_time_and_mints_the_depth_at_the_insert(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = playing_vc(mock_ctx)
        mp = mock_mp()
        mp.enqueue_depth = MagicMock(return_value=7)
        music_bot.get_mp = MagicMock(return_value=mp)
        spy = AsyncMock(
            return_value=QueueObject("https://yt.com/v=1", "Song", mock_ctx.author)
        )
        play_pipeline.queue_source = spy

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        assert spy.await_args is not None
        analytics = spy.await_args.kwargs["analytics"]
        # The message snowflake, NOT time.time(): gateway delivery lag is real
        # time the user waited, and so is the 1-4s resolve that follows.
        assert analytics.queued_at == mock_ctx.message.created_at.timestamp()
        # Not read at the ask — two requests resolving together would both read
        # the same depth. The insert reads it.
        assert analytics.queue_position == 0
        queued = mp.queue_put.await_args.args[0]
        assert queued.analytics.queue_position == 7


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
        vc = paused_vc(mock_ctx)
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
        mock_ctx.voice_client = paused_vc(mock_ctx)
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
        mock_ctx.voice_client = playing_vc(mock_ctx)
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
        mock_ctx.voice_client = paused_vc(mock_ctx)
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
        vc = paused_vc(mock_ctx)
        mock_ctx.voice_client = vc
        mp = self._paused_mp()
        mp.enqueue_depth = MagicMock(return_value=9)
        music_bot.get_mp = MagicMock(return_value=mp)
        qobj = QueueObject("https://yt.com/v=new", "New", mock_ctx.author)
        play_pipeline.queue_source = AsyncMock(return_value=qobj)

        async def _resolve_then_resume(*a: Any, **kw: Any) -> bool:
            vc.is_paused.return_value = False  # user hit -resume mid-extraction
            return True

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(
                YTDL, "prefetch_stream", new=AsyncMock(side_effect=_resolve_then_resume)
            ),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="test")

        mp.interject.assert_not_awaited()
        mp.queue_put.assert_awaited_once_with(qobj)
        assert qobj.interjected is False  # must not trigger replace semantics later
        # Re-minted for the append: the 0 minted for an interjection would claim
        # this song played immediately when it waited behind the whole queue.
        assert qobj.analytics.queue_position == 9

    async def test_resolution_failure_leaves_paused_song_untouched(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Resolution happens before interject, so a failed lookup never stops
        the paused song."""
        vc = paused_vc(mock_ctx)
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
        mock_ctx.voice_client = paused_vc(mock_ctx)
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
            mock_ctx.voice_client = connected_vc(mock_ctx)  # what a real join leaves
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
            mock_ctx.voice_client = connected_vc(mock_ctx)
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
        # No playback hold on the warm path — the gate is already open. It DOES
        # wait on the restore, on the same bound as every placement: a put()
        # landing before restore_entries replays leaves the deque holding this
        # song ahead of entries Redis lists behind it.
        mp.defer_playback.assert_not_called()
        assert (
            mp.wait_for_restore.await_args is not None
            and mp.wait_for_restore.await_args.kwargs["timeout"] == RESTORE_WAIT_SECS
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
            mock_ctx.voice_client = connected_vc(mock_ctx)  # what a real join leaves
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
            mock_ctx.voice_client = connected_vc(mock_ctx)  # what a real join leaves
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
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.COLD_FRONT,
            cog=music_bot,
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
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.COLD_FRONT,
            cog=music_bot,
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
            admit(music_bot, mock_ctx, mp),
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
            mock_ctx.voice_client = connected_vc(mock_ctx)  # what a real join leaves
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
                mock_ctx,
                qobj,
                music_player,
                admit(music_bot, mock_ctx, music_player),
                placement=Placement.COLD_FRONT,
                cog=music_bot,
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
    def _vc(
        *, playing: bool = False, paused: bool = False, ctx: Optional[MagicMock] = None
    ) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = playing
        vc.is_paused.return_value = paused
        vc.is_connected.return_value = True
        return in_authors_channel(vc, ctx)

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
        mock_ctx.voice_client = self._vc(playing=True, ctx=mock_ctx)

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
        mock_ctx.voice_client = self._vc(playing=True, ctx=mock_ctx)
        # The real insert: the depth is minted there, not at the ask.
        play_pipeline.enqueue_single = _REAL_ENQUEUE_SINGLE

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next song"
            )

        queued = seams.mp.queue_put_next.await_args.args[0]
        assert queued.analytics.queue_position == 1
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
        mock_ctx.voice_client = self._vc(paused=True, ctx=mock_ctx)

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
        mock_ctx.voice_client = self._vc(playing=True, ctx=mock_ctx)

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
        mock_ctx.voice_client = (
            self._vc(playing=True, ctx=mock_ctx) if live else connected_vc()
        )
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
        mock_ctx.voice_client = connected_vc(mock_ctx)
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
        mock_ctx.voice_client = connected_vc(mock_ctx)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="-now song")

        seams.queue_source.assert_not_awaited()
        seams.command_error.assert_not_awaited()
        assert "--now" in mock_ctx.send.call_args.kwargs["embed"].description

    async def test_the_flag_with_nothing_behind_it_asks(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        seams = self._wire(music_bot, mock_ctx, live=False)
        mock_ctx.voice_client = connected_vc(mock_ctx)

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
        mock_ctx.voice_client = (
            self._vc(playing=True, ctx=mock_ctx) if live else connected_vc()
        )
        boom = RuntimeError("nope")
        seams.interject.side_effect = boom
        seams.queue_source.side_effect = boom

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now song")

        assert seams.command_error.await_args.kwargs["title"] == title


# ── -play argument parsing ────────────────────────────────────────────────────


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
    def live_vc(self, mock_ctx: MagicMock) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return in_authors_channel(vc, mock_ctx)

    async def test_idle_runs_the_ordinary_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Nothing live to interrupt, so this queues like any other -play — through
        the same body, so the checks, the hooks and the bucket all apply."""
        mp = mock_mp()
        mp.current_song = None
        music_bot.get_mp = MagicMock(return_value=mp)
        mock_ctx.voice_client = connected_vc(mock_ctx)
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
        prefetch = AsyncMock(
            side_effect=lambda *a, **k: order.append("prefetch") or True
        )
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

    async def test_the_prefetch_settles_before_the_place_lock(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """interject()'s own neutralize cancels a prefetch that can be pinned in
        the yt-dlp executor. Run inside _place that wait IS the guild's lock, held
        while every sibling -play in the guild burns its own bound against it and
        reports a Redis outage that never happened."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        order: list[str] = []
        live_mp.settle_prefetch = AsyncMock(
            side_effect=lambda *a, **k: order.append("settle")
        )
        real_place = music_bot._plays.place

        @contextlib.asynccontextmanager
        async def _spy(req: PlayRequest) -> AsyncIterator[PlaceResult]:
            order.append("place")
            async with real_place(req) as verdict:
                yield verdict

        music_bot._plays.place = _spy
        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        assert order == ["settle", "place"]

    async def test_two_concurrent_now_flags_interject_one_at_a_time(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Two `--now` resolving together, driven through the command. Overlapping,
        each would park a resume tail for a song the other already stopped, and one
        play's history row goes with it; only an end-to-end drive proves the
        command takes the lock."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        music_bot._command_error = AsyncMock()
        gate = asyncio.Event()

        async def _resolve(_ctx: Any, _source: Any, **kw: Any) -> QueueObject:
            await gate.wait()
            return QueueObject(
                f"https://yt.com/v={kw['origin']}", kw["origin"], mock_ctx.author
            )

        play_pipeline.queue_source = AsyncMock(side_effect=_resolve)

        inside = 0
        overlapped = False

        async def _interject(*_a: Any, **_k: Any) -> InterjectOutcome:
            nonlocal inside, overlapped
            inside += 1
            overlapped = overlapped or inside > 1
            await asyncio.sleep(0)  # a suspension point inside the hold
            inside -= 1
            return InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=False,
            )

        live_mp.interject = AsyncMock(side_effect=_interject)

        with no_typing("src.commands.play.background_typing"):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(
                        music_bot, mock_ctx, url=f"--now s{n}"
                    )
                )
                for n in (1, 2)
            ]
            await settle()
            assert play_pipeline.queue_source.await_count == 2  # both resolving at once
            gate.set()
            await asyncio.gather(*tasks)

        assert live_mp.interject.await_count == 2  # neither was dropped
        assert not overlapped
        music_bot._command_error.assert_not_awaited()

    async def test_next_settles_the_prefetch_before_the_place_lock(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """--next reaches the same cancel by another road: queue_put_next
        neutralizes the prefetch, and a prefetch pinned in the yt-dlp executor
        cannot be interrupted. Run under the lock it holds the guild's place
        section for the whole extraction, and a --next that outlives the bound is
        cancelled mid-neutralize — not queued, while the notice says it may be."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Next up", mock_ctx.author)
        )

        order: list[str] = []
        live_mp.settle_prefetch = AsyncMock(
            side_effect=lambda *a, **k: order.append("settle")
        )
        real_place = music_bot._plays.place

        @contextlib.asynccontextmanager
        async def _spy(req: PlayRequest) -> AsyncIterator[PlaceResult]:
            order.append("place")
            async with real_place(req) as verdict:
                yield verdict

        music_bot._plays.place = _spy
        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next test"
            )

        assert order == ["settle", "place"]
        live_mp.interject.assert_not_awaited()  # --next never interrupts


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
        vc = paused_vc(mock_ctx)
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
        # Behind the head, through the same insert, so the two land in order.
        assert list(single_call.kwargs["follow_on"]) == tracks[1:]
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
        vc = paused_vc(mock_ctx)
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

        head_call, tail_call = mp.queue_put.await_args_list
        assert head_call.args[0] is tracks[0]
        assert tracks[0].analytics.queue_position == 20
        assert [item.analytics.queue_position for item in tail_call.args[0]] == [21, 22]

    async def test_playlist_interjects_head_first_and_queues_the_rest(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The head interrupts and the rest queue behind it, so the paused song
        comes back after the WHOLE playlist. That is the deliberate call, and the
        confirmation both states it and names the one command that undoes it."""
        mock_ctx.voice_client = paused_vc(mock_ctx)
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
            mock_ctx.voice_client = connected_vc(mock_ctx)  # what a real join leaves
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
            admit(music_bot, mock_ctx, mp),
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
                admit(music_bot, mock_ctx, mp),
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

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=placement,
            cog=music_bot,
        )

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
        mock_ctx.voice_client = connected_vc(mock_ctx)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
            cog=music_bot,
        )

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
        vc = in_authors_channel(MagicMock(spec=discord.VoiceClient), mock_ctx)
        vc.is_paused.return_value = True
        mock_ctx.voice_client = vc

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
            cog=music_bot,
        )

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
        mock_ctx.voice_client = connected_vc(mock_ctx)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
            cog=music_bot,
        )

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
        mock_ctx.voice_client = connected_vc(mock_ctx)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
            cog=music_bot,
        )

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
        mock_ctx.voice_client = connected_vc(mock_ctx)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            placement=Placement.NEXT,
            cog=music_bot,
        )

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
        mock_ctx.voice_client = in_authors_channel(
            MagicMock(spec=discord.VoiceClient), mock_ctx
        )
        mock_ctx.voice_client.is_playing.return_value = True
        qobj = QueueObject("https://yt.com/v=1", "Test Song", mock_ctx.author)
        mp = self._playing_mp(head=qobj)

        await play_pipeline.enqueue_single(
            mock_ctx,
            qobj,
            mp,
            admit(music_bot, mock_ctx, mp),
            note="3 more follow",
            cog=music_bot,
        )

        mp.repin_now_playing.assert_awaited_once()
        mp.build_queued_song_embed.assert_not_called()
        assert "3 more follow" in mock_ctx.send.await_args.kwargs["embed"].description


def _gated_resolve(qobj: QueueObject, gate: asyncio.Event) -> AsyncMock:
    """A queue_source that resolves `qobj` once `gate` is set."""

    async def _resolve(*_a: Any, **_kw: Any) -> QueueObject:
        await gate.wait()
        return qobj

    return AsyncMock(side_effect=_resolve)


def _holding_mp() -> MagicMock:
    """mock_mp() whose defer_playback really counts holds, for the cold-start
    tests: _abandon_cold_start reads playback_holds, and two participants must
    see each other's."""
    mp = mock_mp()
    mp.playback_holds = 0

    @contextlib.asynccontextmanager
    async def _hold() -> AsyncIterator[None]:
        mp.playback_holds += 1
        try:
            yield
        finally:
            mp.playback_holds -= 1

    mp.defer_playback = MagicMock(side_effect=_hold)
    return mp


async def _stalled_put(*_a: Any, **_k: Any) -> None:
    """A put against a Redis that accepts and never answers."""
    await asyncio.sleep(5)


class TestResolveThenPlace:
    """Requests resolve together and insert one at a time, each where the queue
    is when its own resolve finishes."""

    def _warm(self, music_bot: MusicBot, mock_ctx: MagicMock) -> MagicMock:
        mock_ctx.voice_client = playing_vc(mock_ctx)
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_requests_resolve_concurrently(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()
        play_pipeline.queue_source = _gated_resolve(song(1, mock_ctx), gate)

        with no_typing("src.commands.play.background_typing"):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await settle()
            # Both are inside the resolver; neither has placed.
            assert play_pipeline.queue_source.await_count == 2
            mp.queue_put.assert_not_awaited()
            assert len(music_bot._plays._guilds[play_key(mock_ctx)].inflight) == 2

            gate.set()
            await asyncio.gather(*tasks)

        assert mp.queue_put.await_count == 2
        assert not music_bot._plays._guilds

    async def test_a_short_resolve_places_before_a_long_one(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Decision 2.1: no ticket order. The song asked second lands first when
        the collection asked first is still extracting."""
        mp = self._warm(music_bot, mock_ctx)
        slow_gate = asyncio.Event()
        slow, fast = song(1, mock_ctx), song(2, mock_ctx)

        async def _resolve(_ctx: Any, _source: Any, **_kw: Any) -> QueueObject:
            if _kw["origin"] == "slow":
                await slow_gate.wait()
                return slow
            return fast

        play_pipeline.queue_source = AsyncMock(side_effect=_resolve)

        with no_typing("src.commands.play.background_typing"):
            first = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="slow")
            )
            await settle()
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="fast")
            assert mp.queue_put.await_args.args[0] is fast
            slow_gate.set()
            await first

        assert [c.args[0] for c in mp.queue_put.await_args_list] == [fast, slow]

    async def test_the_reply_is_sent_outside_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A 429 on the channel must not hold every other -play in the guild: the
        confirmation goes out after the lock is released."""
        mp = self._warm(music_bot, mock_ctx)
        mp.queue.qsize = MagicMock(return_value=3)  # so a confirmation is sent
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        send_gate = asyncio.Event()
        sends = 0

        async def _blocked_send(**_kw: Any) -> None:
            nonlocal sends
            sends += 1
            if sends == 1:
                await send_gate.wait()

        mock_ctx.send = AsyncMock(side_effect=_blocked_send)

        with no_typing("src.commands.play.background_typing"):
            first = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            )
            await settle()
            assert sends == 1  # stuck in its reply
            assert not music_bot._plays._guilds[play_key(mock_ctx)].lock.locked()
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="b")
            assert mp.queue_put.await_count == 2  # placed behind a blocked reply
            send_gate.set()
            await first

    async def test_the_confirmation_is_rendered_outside_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Only the put and the depth on it belong under the hold. Rendering an
        "Est. playing at" walks the whole queue, and the hold is shared by every
        -play in the guild — under it, one long queue's walk is time every sibling
        spends waiting for a lock it is not using."""
        mp = self._warm(music_bot, mock_ctx)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        held: list[bool] = []

        def _eta(*_a: Any, **_k: Any) -> str:
            plays = music_bot._plays._guilds[play_key(mock_ctx)]
            held.append(plays.lock.locked())
            return "soon"

        mp.build_queued_song_embed = MagicMock(side_effect=_eta)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_awaited_once()  # it did place, so the ETA was rendered
        assert held == [False]

    async def test_a_refusal_is_sent_outside_the_lock(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mock_ctx.author.voice = None  # left during the resolve
        send_gate = asyncio.Event()

        async def _blocked_send(**_k: Any) -> None:
            await send_gate.wait()

        mock_ctx.send = AsyncMock(side_effect=_blocked_send)

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            )
            await settle()
            mock_ctx.send.assert_awaited_once()
            assert not music_bot._plays._guilds[play_key(mock_ctx)].lock.locked()
            send_gate.set()
            await task

        mp.queue_put.assert_not_awaited()

    async def test_a_served_request_records_resolve_and_place_wait(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._warm(music_bot, mock_ctx)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))

        with no_typing("src.commands.play.background_typing"), recording_span() as span:
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        recorded = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert recorded["play.inflight"] == 1
        assert recorded["play.resolve_secs"] >= 0
        assert recorded["play.place_wait_secs"] >= 0
        assert "play.declined" not in recorded
        assert "play.dropped_by" not in recorded

    async def test_the_place_timeout_reports_and_releases(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A Redis stall inside the put: the request reports, the lock is free
        for the next one, and nothing else is sent."""
        mp = self._warm(music_bot, mock_ctx)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mp.queue_put = AsyncMock(side_effect=_stalled_put)

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mock_ctx.send.assert_awaited_once()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "queue is busy" in text
        # queue_put appends to the deque BEFORE it awaits the mirror, so a stall
        # here may well have queued the song. Claiming it did not, and inviting a
        # retry, is what mints the duplicate.
        assert "wasn't queued" not in text
        assert "-queue" in text
        assert not music_bot._plays._guilds  # retired, lock gone with it

    async def test_a_cold_start_that_stalls_at_the_insert_is_abandoned(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.voice_client = None
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        play_cmd.abandon_cold_start = AsyncMock()
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mp.queue_put_front = AsyncMock(side_effect=_stalled_put)

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        play_cmd.abandon_cold_start.assert_awaited_once()
        assert "queue is busy" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_a_stalled_tail_does_not_disconnect_what_its_siblings_queued(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The hold is one Redis round trip long, so a guild bursting to the
        admission cap serializes that many and the last request can spend its whole
        budget waiting. Torn down there, one refusal disconnects a joined session
        and takes every song the other fifteen placed with it."""
        mock_ctx.voice_client = None
        mp = mock_mp(qsize=3)  # siblings placed while this one waited
        music_bot.get_mp = MagicMock(return_value=mp)
        play_cmd.abandon_cold_start = AsyncMock()
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mp.queue_put_front = AsyncMock(side_effect=_stalled_put)

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        play_cmd.abandon_cold_start.assert_not_awaited()
        assert "queue is busy" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_a_refused_tail_does_not_disconnect_either(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The other late exit, reached by a verdict rather than a clock. Same
        session, same siblings, same reason not to take it down."""
        mock_ctx.voice_client = None
        mp = mock_mp(qsize=3)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_cmd.abandon_cold_start = AsyncMock()
        gate = asyncio.Event()

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_slow)

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            )
            await settle()
            mock_ctx.author.voice = None  # left while it resolved
            gate.set()
            await task

        mp.queue_put_front.assert_not_awaited()
        play_cmd.abandon_cold_start.assert_not_awaited()

    async def test_a_connection_lost_mid_resolve_is_still_torn_down(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The other half of the guard. A queue with songs in it is only a reason
        to stay if there is still a connection to play them over: released with
        none, the loop fails its `vc` assertion once per restored song while Redis
        keeps every entry, and the next restore does it again."""
        mock_ctx.voice_client = None
        mp = mock_mp(qsize=3)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_cmd.abandon_cold_start = AsyncMock()
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mp.queue_put_front = AsyncMock(side_effect=_stalled_put)

        async def _join(*_a: Any, **_k: Any) -> None:
            vc = connected_vc(mock_ctx)
            # Connected for the post-join check, gone by the time the stall asks:
            # a kick between the handshake and the lock.
            vc.is_connected = MagicMock(side_effect=[True, False])
            mock_ctx.voice_client = vc

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        play_cmd.abandon_cold_start.assert_awaited_once()

    async def test_a_stall_waiting_for_the_lock_says_the_song_is_absent(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The timeout is outer to the lock: a request parked behind a stalled sibling
        gives up on its own clock, and having never acquired the lock it wrote
        nothing, so this half may say so (the put-stall half may not)."""
        mp = self._warm(music_bot, mock_ctx)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        plays = _GuildPlays()
        music_bot._plays._guilds[play_key(mock_ctx)] = plays
        await plays.lock.acquire()  # a sibling holds it and never returns

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_not_awaited()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "queue is busy" in text
        assert "wasn't queued" in text  # nothing was written, so it may say so
        plays.lock.release()

    async def test_an_author_who_left_during_the_resolve_does_not_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)

        async def _leave(*_a: Any, **_kw: Any) -> QueueObject:
            mock_ctx.author.voice = None
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_leave)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_not_awaited()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "not connected to a voice channel" in text

    async def test_queue_position_is_the_depth_at_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Decision 2.3. Two asks read the same depth; two inserts do not."""
        mp = self._warm(music_bot, mock_ctx)
        placed: list[QueueObject] = []

        async def _put(obj: Any, **_k: Any) -> None:
            placed.append(obj)

        mp.queue_put = AsyncMock(side_effect=_put)
        mp.enqueue_depth = MagicMock(side_effect=lambda: 4 + len(placed))
        gate = asyncio.Event()
        songs = iter([song(1, mock_ctx), song(2, mock_ctx)])

        async def _resolve(*_a: Any, **_kw: Any) -> QueueObject:
            await gate.wait()
            return next(songs)

        play_pipeline.queue_source = AsyncMock(side_effect=_resolve)

        with no_typing("src.commands.play.background_typing"):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await settle()
            gate.set()
            await asyncio.gather(*tasks)

        assert [q.analytics.queue_position for q in placed] == [4, 5]

    async def test_now_interjects_without_waiting_for_a_collection(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The lane plan's §12 row 3, removed rather than accepted: --now lands
        while the collection asked before it is still resolving."""
        mp = self._warm(music_bot, mock_ctx)
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Old",
                resume_position=10,
                was_paused=False,
                returns_paused=False,
            )
        )
        gate = asyncio.Event()
        play_pipeline.queue_source = _gated_resolve(song(1, mock_ctx), gate)
        play_pipeline._resolve_interjection_source = AsyncMock(
            return_value=(song(2, mock_ctx), [])
        )

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            collection = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="list")
            )
            await settle()
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now x")
            mp.interject.assert_awaited_once()
            assert not collection.done()
            gate.set()
            await collection

    async def test_a_cold_start_records_the_join_wait_separately(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Folded into play.resolve_secs, the cold-start half of any "split the
        resolve by ytdl.flat" reading is a voice handshake wearing an extraction's
        name — and a cold start is a large share of plays."""
        mock_ctx.voice_client = None
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        play_pipeline.enqueue_single = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mock_mp())

        loop = asyncio.get_event_loop()
        join_task = loop.create_future()
        join_task.set_result(None)

        def fake_create_task(coro: Coroutine[Any, Any, Any]) -> asyncio.Future:
            coro.close()
            mock_ctx.voice_client = connected_vc(mock_ctx)
            return join_task

        with (
            no_typing("src.commands.play.background_typing"),
            recording_span() as span,
            patch("asyncio.create_task", side_effect=fake_create_task),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        recorded = {c.args[0]: c.args[1] for c in span.set_attribute.call_args_list}
        assert recorded["play.resolve_secs"] >= 0
        assert recorded["play.join_wait_secs"] >= 0

    def test_the_place_bound_outlives_the_start_write_it_waits_behind(self) -> None:
        """A song start holds the queue mutex for up to _START_WRITE_TIMEOUT against a
        Redis that accepts and stalls, and every placing request in that guild waits
        behind it. Equal bounds expire together, so the placer would report a stall at
        the instant the mutex it wanted was about to free."""
        assert PLACE_TIMEOUT_SECS > _START_WRITE_TIMEOUT

    async def test_a_place_waiting_out_a_held_lock_lands_rather_than_stalls(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """What that margin buys, at the real bound: a request parked on the lock for
        as long as a stalled start can hold it still places when the lock frees."""
        mp = mock_mp()
        holder = admit(music_bot, mock_ctx, mp)
        waiter = admit(music_bot, mock_ctx, mp)
        # Event-driven rather than timed: the assertion is that the waiter did not
        # give up, and a sleep long enough to prove that would be a slow flake.
        finish_the_put = asyncio.Event()

        async def _hold() -> None:
            async with music_bot._plays.place(holder):
                await finish_the_put.wait()

        held = asyncio.ensure_future(_hold())
        await asyncio.sleep(0)  # the holder takes the lock first

        async def _wait_and_place() -> PlaceResult:
            async with music_bot._plays.place(waiter) as result:
                pass
            return result

        parked = asyncio.ensure_future(_wait_and_place())
        for _ in range(3):
            await asyncio.sleep(0)
        assert not parked.done()  # genuinely blocked on the lock

        finish_the_put.set()
        await held
        assert (await parked).placed
        assert waiter.placed


class TestPlacementRevalidation:
    """What the insert re-checks, and what a command can still take back."""

    def _resolving(
        self, music_bot: MusicBot, mock_ctx: MagicMock, gate: asyncio.Event
    ) -> MagicMock:
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_slow)
        return mp

    async def test_a_now_whose_author_changed_channels_does_not_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """validate_commands gates --now on the bot's own channel, because it stops
        what that channel is hearing. The re-check after the resolve has to carry
        the same queue_control, or walking to another channel during a 1-99s
        extraction buys an exemption the command never had."""
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)
        mp.current_song = None

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now x")
            )
            await settle()
            mock_ctx.author.voice.channel = MagicMock(spec=discord.VoiceChannel)
            gate.set()
            await task

        mp.queue_put_next.assert_not_awaited()
        mp.interject.assert_not_called()  # bare Mock: no await-flavoured assert
        assert (
            "already being used" in mock_ctx.send.await_args.kwargs["embed"].description
        )

    async def test_a_plain_play_keeps_its_exemption(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Queueing into a session running elsewhere costs its listeners nothing,
        so -play alone is exempt at dispatch and must stay exempt here."""
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="x")
            )
            await settle()
            mock_ctx.author.voice.channel = MagicMock(spec=discord.VoiceChannel)
            gate.set()
            await task

        mp.queue_put.assert_awaited_once()

    async def test_remove_reaches_a_request_still_resolving(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The window this branch opened: a resolve runs to 99s, and the argument
        the user is taking back is the one they just typed. Matched by origin, the
        way the queue matches it."""
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[], positions=[], mode=RemoveMode.RESOLVED
            )
        )

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="Bad Song")
            )
            await settle()
            await command_callback(MusicBot.remove)(
                music_bot, mock_ctx, needle="bad song"
            )
            gate.set()
            await task

        mp.queue_put.assert_not_awaited()
        said = [
            c.kwargs["embed"].description or ""
            for c in mock_ctx.send.await_args_list
            if c.kwargs.get("embed") is not None
        ]
        assert any("still being looked up" in t for t in said), said
        assert any("`-remove` ran while it was resolving" in t for t in said), said

    async def test_remove_leaves_an_unrelated_request_alone(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        gate = asyncio.Event()
        mp = self._resolving(music_bot, mock_ctx, gate)
        mp.queue_remove = AsyncMock(
            return_value=RemoveOutcome(
                removed=[], positions=[], mode=RemoveMode.RESOLVED
            )
        )

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="Keep This")
            )
            await settle()
            await command_callback(MusicBot.remove)(
                music_bot, mock_ctx, needle="something else"
            )
            gate.set()
            await task

        mp.queue_put.assert_awaited_once()

    async def test_a_placed_request_is_not_reported_as_dropped(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """PlayRegistry.retire runs in play()'s finally, AFTER the confirmation is sent, so
        a request whose song is already queued is still in the registry. Stamping it
        names it in -clear's own dropped field beside the song it just cleared."""
        mp = mock_mp(3)  # something queued, so a confirmation is actually sent
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        gate = asyncio.Event()

        async def _slow_send(*_a: Any, **_k: Any) -> MagicMock:
            await gate.wait()
            return MagicMock()

        # Blocked BEFORE the task starts, so it parks after the put and before
        # the retire: exactly the window where a placed request is still listed.
        mock_ctx.send = AsyncMock(side_effect=_slow_send)
        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="queued one")
            )
            await settle()
            mp.queue_put.assert_awaited_once()  # it placed
            assert music_bot._plays._guilds[
                play_key(mock_ctx)
            ].inflight  # and is listed
            dropped = music_bot._plays.inflight(play_key(mock_ctx), "clear")
            gate.set()
            await task

        assert dropped == []  # no command can take it back now

    async def test_a_cold_start_refused_at_the_lock_is_torn_down(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The join already put the bot in the channel. Every other exit tears it
        down; without this one it sits in an empty channel until the 300s idle."""
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()
        play_cmd.abandon_cold_start = AsyncMock()
        gate = asyncio.Event()

        async def _join(*_a: Any, **_k: Any) -> None:
            mock_ctx.voice_client = connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_slow)
        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="x")
            )
            await settle()
            mock_ctx.author.voice = None  # left while it resolved
            gate.set()
            await task

        mp.queue_put_front.assert_not_awaited()
        play_cmd.abandon_cold_start.assert_awaited()

    async def test_a_failed_reaction_does_not_fail_a_queued_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Send Messages without Add Reactions is a common split, and by the time
        the reaction runs the song is IN the queue. Raising there renders "Failed
        to queue song" over a song that plays."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mock_ctx.message.add_reaction = AsyncMock(
            side_effect=discord.Forbidden(MagicMock(status=403), "no reactions")
        )

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_awaited_once()
        for call_ in mock_ctx.send.await_args_list:
            embed = call_.kwargs.get("embed")
            assert embed is None or "Failed to queue" not in (embed.title or "")

    async def test_a_body_timeout_is_not_reported_as_a_stalled_queue(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """_place catches TimeoutError, which since 3.11 IS asyncio.TimeoutError.
        Anything inside the body raising one — a nested wait_for, a guard added
        later — would otherwise be reclassified as a Redis stall."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mp.queue_put = AsyncMock(side_effect=TimeoutError("someone else's deadline"))

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        text = mock_ctx.send.await_args.kwargs["embed"].description or ""
        assert "queue is busy" not in text
        assert "Failed to queue song" in (
            mock_ctx.send.await_args.kwargs["embed"].title or ""
        )


class TestPlayShowsTyping:
    async def test_the_body_runs_inside_the_typing_indicator(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Every other -play test patches background_typing away, so deleting the
        `async with` leaves the suite green and the bot silent for the 1-99s a
        resolve takes."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        entered: list[str] = []

        @contextlib.asynccontextmanager
        async def _typing(_ctx: Any) -> AsyncIterator[None]:
            entered.append("typing")
            yield

        with patch("src.commands.play.background_typing", new=_typing):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        assert entered == ["typing"]


class TestSpanDecoratorsNameTheirFunction:
    """A helper inserted between a decorator and the function it was written for
    silently inherits its span: bot.enqueue_playlist once measured a --next-only
    stream warm while the playlist enqueue itself went untraced."""

    async def test_every_tracer_decorator_wraps_the_function_it_names(self) -> None:
        import inspect
        import re

        import src.musicbot as module

        lines = inspect.getsource(module).splitlines()
        spans = [
            (i, m.group(1))
            for i, line in enumerate(lines)
            if (m := re.search(r'start_as_current_span\("bot\.([a-z_]+)"\)', line))
        ]
        assert len(spans) > 15, "the decorators moved; re-anchor this test"
        for i, name in spans:
            following = next(line.strip() for line in lines[i + 1 :] if line.strip())
            defined = re.match(r"(?:async )?def (\w+)", following)
            assert defined, (name, following)
            assert defined.group(1) in (name, f"_{name}"), (name, defined.group(1))


class TestPlaceRefuses:
    """The four checks a resolved request can fail at the lock, and what each
    side says about it."""

    def _warm(self, music_bot: MusicBot, mock_ctx: MagicMock) -> MagicMock:
        mock_ctx.voice_client = playing_vc(mock_ctx)
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        # -clear on a MagicMock player: bump the generation the way clear() does.
        async def _clear() -> list[Any]:
            mp.queue.generation += 1
            return []

        mp.queue_clear = AsyncMock(side_effect=_clear)
        return mp

    async def test_clear_during_the_resolve_drops_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()
        play_pipeline.queue_source = _gated_resolve(song(1, mock_ctx), gate)

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="my song")
            )
            await settle()
            await command_callback(MusicBot.clear)(music_bot, mock_ctx)
            gate.set()
            await task

        mp.queue_put.assert_not_awaited()
        # Two reports: -clear names the request it dropped, and the request
        # names the command that dropped it.
        descriptions = [
            c.kwargs["embed"].description for c in mock_ctx.send.await_args_list
        ]
        clear_embed = mock_ctx.send.await_args_list[0].kwargs["embed"]
        assert clear_embed.fields[0].name == "1 play request dropped"
        assert "my song" in clear_embed.fields[0].value
        assert any("`-clear` ran while it was resolving" in d for d in descriptions)

    async def test_clear_reports_even_when_the_queue_was_already_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        admit(music_bot, mock_ctx, mp, query="pending")

        await command_callback(MusicBot.clear)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "already empty" not in (embed.description or "") or embed.fields
        # The title counts what the command actually took. "Queue cleared — 0 songs
        # removed" over a field naming a dropped request contradicts itself.
        assert embed.title == "Cleared — 1 play request dropped"
        assert embed.fields[0].name == "1 play request dropped"

    async def test_clear_with_nothing_to_drop_still_says_already_empty(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        self._warm(music_bot, mock_ctx)
        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        assert "already empty" in mock_ctx.send.await_args.kwargs["embed"].description

    async def test_stop_during_the_resolve_drops_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()
        play_pipeline.queue_source = _gated_resolve(song(1, mock_ctx), gate)

        async def _cleanup(_guild: Any) -> None:
            mp.retired = True  # what cleanup() stamps before any await

        music_bot.cleanup = AsyncMock(side_effect=_cleanup)
        mock_ctx.message.add_reaction = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="my song")
            )
            await settle()
            with patch("discord.utils.get", return_value=playing_vc()):
                await command_callback(MusicBot.stop)(music_bot, mock_ctx)
            gate.set()
            await task

        mp.queue_put.assert_not_awaited()
        embeds = [c.kwargs["embed"] for c in mock_ctx.send.await_args_list]
        assert embeds[0].title == "Stopped"
        assert embeds[0].fields[0].name == "1 play request dropped"
        assert any(
            "`-stop` ran while it was resolving" in (e.description or "")
            for e in embeds
        )

    async def test_stop_drops_a_cold_start_before_there_is_a_client_to_find(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Unconditional: the voice-client check that gates the teardown does
        not gate the report, since a cold start is resolving before any client
        exists."""
        mp = self._warm(music_bot, mock_ctx)
        admit(music_bot, mock_ctx, mp, query="cold one")
        music_bot.cleanup = AsyncMock()

        with patch("discord.utils.get", return_value=None):
            await command_callback(MusicBot.stop)(music_bot, mock_ctx)

        music_bot.cleanup.assert_not_awaited()
        assert mock_ctx.send.await_args.kwargs["embed"].fields[0].name == (
            "1 play request dropped"
        )

    async def test_a_request_stop_reported_dropped_does_not_then_place(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """On this leg -stop retires no player and bumps no generation — there is
        no voice client to clean up — so the stamp is the ONLY invalidation. Left
        as a label the channel is told the request was dropped and it queues,
        joins and plays anyway."""
        mp = self._warm(music_bot, mock_ctx)
        gate = asyncio.Event()

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_slow)
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="cold one")
            )
            await settle()
            with patch("discord.utils.get", return_value=None):
                await command_callback(MusicBot.stop)(music_bot, mock_ctx)
            gate.set()
            await task

        music_bot.cleanup.assert_not_awaited()  # nothing was retired
        mp.queue_put.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        said = mock_ctx.send.await_args.kwargs["embed"].description or ""
        assert "`-stop` ran while it was resolving" in said

    async def test_a_kick_during_the_resolve_drops_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """No command stamped it, so the message names the session, not a
        command."""
        mp = self._warm(music_bot, mock_ctx)

        async def _kicked(*_a: Any, **_kw: Any) -> QueueObject:
            mp.retired = True
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_kicked)

        with no_typing("src.commands.play.background_typing"), recording_span() as span:
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_not_awaited()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "the session ended" in text
        span.set_attribute.assert_any_call("play.dropped_by", "session")

    @pytest.mark.parametrize("url", ["song", "--next song", "--now song"])
    async def test_a_dropped_request_never_places(
        self, music_bot: MusicBot, mock_ctx: MagicMock, url: str
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        mp.current_song = MagicMock()
        mp.interject = AsyncMock()

        async def _cleared(*_a: Any, **_kw: Any) -> Any:
            mp.queue.generation += 1
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_cleared)

        async def _cleared_pair(*_a: Any, **_kw: Any) -> Any:
            mp.queue.generation += 1
            return song(1, mock_ctx), []

        play_pipeline._resolve_interjection_source = AsyncMock(
            side_effect=_cleared_pair
        )

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url=url)

        mp.queue_put.assert_not_awaited()
        mp.queue_put_next.assert_not_awaited()
        mp.queue_put_front.assert_not_awaited()
        mp.interject.assert_not_awaited()
        assert (
            "queue was cleared" in mock_ctx.send.await_args.kwargs["embed"].description
        )

    async def test_a_dropped_collection_never_places(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = self._warm(music_bot, mock_ctx)
        tracks = [song(1, mock_ctx), song(2, mock_ctx)]

        async def _cleared(*_a: Any, **_kw: Any) -> ResolvedYoutubePlaylist:
            mp.queue.generation += 1
            return ResolvedYoutubePlaylist(tracks)

        play_pipeline.queue_source = AsyncMock(side_effect=_cleared)

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="https://www.youtube.com/playlist?list=PLx"
            )

        mp.queue_put.assert_not_awaited()
        mock_ctx.send.assert_awaited_once()
        assert (
            "queue was cleared" in mock_ctx.send.await_args.kwargs["embed"].description
        )

    async def test_an_interjection_that_stalls_at_the_lock_reports_too(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The interject route has no gate hold to unwind; it reports the same
        notice from _play's own handler."""
        mp = self._warm(music_bot, mock_ctx)
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(side_effect=_stalled_put)
        play_pipeline._resolve_interjection_source = AsyncMock(
            return_value=(song(1, mock_ctx), [])
        )

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now x")

        assert "queue is busy" in mock_ctx.send.await_args.kwargs["embed"].description
        assert not music_bot._plays._guilds

    async def test_a_request_placed_before_the_clear_is_an_ordinary_song(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The boundary: placed means in the queue, and -clear's generation
        bump is what a later placer refuses on, not an earlier one."""
        mp = self._warm(music_bot, mock_ctx)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
        mp.queue_put.assert_awaited_once()

        await command_callback(MusicBot.clear)(music_bot, mock_ctx)
        assert not music_bot._plays._guilds  # nothing in flight to name


class TestColdStartSingleflight:
    """One -join per guild, shared by every request that found no voice client."""

    def _cold(
        self, music_bot: MusicBot, mock_ctx: MagicMock, *, join_gate: asyncio.Event
    ) -> MagicMock:
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()

        async def _join(*_a: Any, **_k: Any) -> None:
            await join_gate.wait()
            mock_ctx.voice_client = connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)
        return mp

    async def _two_cold_starts(
        self, music_bot: MusicBot, mock_ctx: MagicMock, join_gate: asyncio.Event
    ) -> tuple[MagicMock, list[asyncio.Task[None]]]:
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        songs = iter([song(1, mock_ctx), song(2, mock_ctx)])
        play_pipeline.queue_source = AsyncMock(
            side_effect=lambda *_a, **_k: next(songs)
        )
        tasks = [
            asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
            )
            for n in (1, 2)
        ]
        await settle()
        return mp, tasks

    async def test_two_cold_starts_share_one_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        join_gate = asyncio.Event()
        with no_typing("src.commands.play.background_typing"):
            mp, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            mock_ctx.invoke.assert_awaited_once()
            assert mp.playback_holds == 2
            join_gate.set()
            await asyncio.gather(*tasks)

        mock_ctx.invoke.assert_awaited_once()
        assert mp.queue_put_front.await_count == 2  # both ahead of the leftovers
        assert mp.playback_holds == 0
        assert not music_bot._plays._guilds

    async def _two_cold_starts_onto_a_failed_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock, join_gate: asyncio.Event
    ) -> tuple[MagicMock, list[asyncio.Task[None]]]:
        """Both requests share one join that leaves no voice client. join()
        swallows its own error into _command_error, so the task completes and the
        waiters see only its absence."""
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()

        async def _failed_join(*_a: Any, **_k: Any) -> None:
            await join_gate.wait()  # and no voice client, ever

        mock_ctx.invoke = AsyncMock(side_effect=_failed_join)
        songs = iter([song(1, mock_ctx), song(2, mock_ctx)])
        play_pipeline.queue_source = AsyncMock(
            side_effect=lambda *_a, **_k: next(songs)
        )
        tasks = [
            asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
            )
            for n in (1, 2)
        ]
        await settle()
        return mp, tasks

    async def test_a_cancelled_play_still_retires_its_slot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """_play catches Exception, so play() only unwinds abnormally on a
        BaseException — which no test reaching the registry produced. Each leak
        costs a slot, and PLAY_INFLIGHT_MAX of them decline the guild until
        restart."""
        join_gate = asyncio.Event()
        with no_typing("src.commands.play.background_typing"):
            _, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            tasks[0].cancel()
            await settle()
            join_gate.set()
            await tasks[1]
            await settle()

        assert not music_bot._plays._guilds

    async def test_a_stale_join_callback_does_not_null_its_replacement(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A cancelled join settles a tick later and its done-callback runs with a
        replacement already installed. Clearing the slot unconditionally there
        hands the NEXT cold start an empty slot, and it spawns a second concurrent
        -join for the same guild — the one thing the singleflight exists to stop."""
        mock_ctx.voice_client = None
        mp = _holding_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._restore_tasks = set()
        gate = asyncio.Event()

        async def _join(*_a: Any, **_k: Any) -> None:
            await gate.wait()
            mock_ctx.voice_client = connected_vc(mock_ctx)

        mock_ctx.invoke = AsyncMock(side_effect=_join)
        key = play_key(mock_ctx)
        plays = music_bot._plays._guilds.setdefault(key, _GuildPlays())
        first = self._make_join(music_bot, mock_ctx, plays)
        first.cancel()
        second = self._make_join(music_bot, mock_ctx, plays)
        await settle()  # the cancelled task's callback has now run

        assert plays.join is second
        gate.set()
        with contextlib.suppress(Exception):
            await second

    @staticmethod
    def _make_join(
        music_bot: MusicBot, mock_ctx: MagicMock, plays: _GuildPlays
    ) -> asyncio.Task[Any]:
        req = PlayRequest(
            ctx=mock_ctx,
            guild_id=play_key(mock_ctx),
            query="x",
            mp=music_bot.get_mp(mock_ctx),
            generation=0,
            mode=PlayMode.NORMAL,
        )
        plays.inflight.append(req)
        join, _ = music_bot._plays.cold_join(
            req,
            joiner=lambda: mock_ctx.invoke(music_bot.join),
            tracked=music_bot._restore_tasks,
        )
        return join

    async def test_a_request_arriving_mid_handshake_joins_the_cold_path(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """discord.py registers the voice client BEFORE the handshake completes, so
        for most of a join an arriving request sees one. Read as warm it takes no
        gate hold and is invisible to _abandon_cold_start's census, and a creator
        whose resolve then fails tears the player down under it."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        songs = iter([song(1, mock_ctx), song(2, mock_ctx)])
        play_pipeline.queue_source = AsyncMock(
            side_effect=lambda *_a, **_k: next(songs)
        )

        with no_typing("src.commands.play.background_typing"):
            first = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="s1")
            )
            await settle()
            # Mid-handshake: the client exists, the join has not finished.
            mock_ctx.voice_client = connected_vc(mock_ctx)
            second = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="s2")
            )
            await settle()
            assert mp.playback_holds == 2, "the second request took no gate hold"
            join_gate.set()
            await asyncio.gather(first, second)

        mock_ctx.invoke.assert_awaited_once()  # still one join

    async def test_a_request_that_only_waited_reports_the_failed_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The P0: join() reports into the creator's context, so a request that
        merely awaited that join used to return None and say nothing at all."""
        join_gate = asyncio.Event()
        with no_typing("src.commands.play.background_typing"):
            mp, tasks = await self._two_cold_starts_onto_a_failed_join(
                music_bot, mock_ctx, join_gate
            )
            join_gate.set()
            await asyncio.gather(*tasks)

        mp.queue_put_front.assert_not_awaited()
        said = [
            call.kwargs["embed"].description
            for call in mock_ctx.send.await_args_list
            if call.kwargs.get("embed") is not None
        ]
        told = [text for text in said if "Couldn't join the voice channel" in text]
        # Exactly one: the waiter speaks for itself, the creator does not repeat
        # what join() already said in this channel.
        assert len(told) == 1, said
        assert "s2" in told[0]  # and it names the song that was dropped

    async def test_a_failed_join_names_itself_on_the_span(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """join() swallows its own error, so without this the drop leaves no
        trace at all: no status, no attribute, no resolve_secs."""
        join_gate = asyncio.Event()
        with no_typing("src.commands.play.background_typing"), recording_span() as span:
            _, tasks = await self._two_cold_starts_onto_a_failed_join(
                music_bot, mock_ctx, join_gate
            )
            join_gate.set()
            await asyncio.gather(*tasks)

        assert ("play.dropped_by", "join_failed") in [
            recorded.args for recorded in span.set_attribute.call_args_list
        ]

    async def test_a_join_that_lands_reports_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The notice rides the failure, not the wait: a shared join that works
        leaves both requests on the ordinary confirmation path."""
        join_gate = asyncio.Event()
        with no_typing("src.commands.play.background_typing"):
            mp, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            join_gate.set()
            await asyncio.gather(*tasks)

        assert mp.queue_put_front.await_count == 2
        for call_ in mock_ctx.send.await_args_list:
            embed = call_.kwargs.get("embed")
            if embed is not None:
                assert "Couldn't join the voice channel" not in (
                    embed.description or ""
                )

    async def test_a_stall_records_itself_on_the_span_and_the_log(
        self, music_bot: MusicBot, mock_ctx: MagicMock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """_PlaceStalled becomes a notice at both catch sites, so it never reaches
        _command_error's log-and-record: without this the failure most likely to be
        systemic — a Redis that accepts and stalls — leaves no operator signal."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        mp.queue_put = AsyncMock(side_effect=_stalled_put)

        with (
            no_typing("src.commands.play.background_typing"),
            recording_span() as span,
            patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        recorded = [call.args for call in span.set_attribute.call_args_list]
        assert ("play.verdict", "stalled") in recorded
        # Set inside the lock on every other path, so a request that never
        # acquired it would otherwise report no wait at all.
        assert any(name == "play.place_wait_secs" for name, _ in recorded)
        span.record_exception.assert_called()
        assert any("Place stalled" in r.message for r in caplog.records)

    @pytest.mark.parametrize("in_voice, wire", [(True, "place"), (False, "voice")])
    async def test_the_span_names_the_verdict_even_when_no_command_dropped_it(
        self, music_bot: MusicBot, mock_ctx: MagicMock, in_voice: bool, wire: str
    ) -> None:
        """dropped_by names a COMMAND; a voice refusal has none, so without a
        verdict attribute it is indistinguishable from a successful place."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))
        if not in_voice:
            mock_ctx.author.voice = None

        with no_typing("src.commands.play.background_typing"), recording_span() as span:
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        assert ("play.verdict", wire) in [
            call.args for call in span.set_attribute.call_args_list
        ]

    async def test_a_dropped_request_names_the_song_it_lost(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Three resolving requests get three of these; identical text leaves the
        author unable to tell which one died."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        gate = asyncio.Event()

        async def _slow(*_a: Any, **_k: Any) -> QueueObject:
            await gate.wait()
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_slow)
        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(
                    music_bot, mock_ctx, url="never gonna give you up"
                )
            )
            await settle()
            mp.retired = True  # -stop, a kick, the alone-watchdog
            gate.set()
            await task

        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "the session ended" in text
        assert "never gonna give you up" in text

    @pytest.mark.parametrize("cancelled", [0, 1], ids=["creator", "participant"])
    async def test_a_participant_cannot_cancel_the_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock, cancelled: int
    ) -> None:
        join_gate = asyncio.Event()
        play_cmd.abandon_cold_start = AsyncMock()
        with no_typing("src.commands.play.background_typing"):
            mp, tasks = await self._two_cold_starts(music_bot, mock_ctx, join_gate)
            tasks[cancelled].cancel()
            await settle()
            join = music_bot._plays._guilds[play_key(mock_ctx)].join
            assert join is not None and not join.cancelled()
            join_gate.set()
            await tasks[1 - cancelled]

        assert tasks[cancelled].cancelled()
        assert mp.queue_put_front.await_count == 1  # the survivor placed

    async def test_alone_a_failed_resolve_cancels_its_own_join(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The single-request rule is unchanged: nobody else waits on the join,
        and a teardown under a still-connecting join leaves join() to rebuild
        the player it then finds missing."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        play_pipeline.queue_source = AsyncMock(side_effect=RuntimeError("no such song"))
        play_cmd.abandon_cold_start = AsyncMock()
        music_bot._command_error = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")
            await settle()

        play_cmd.abandon_cold_start.assert_awaited_once()
        assert not music_bot._plays._guilds  # the cancelled join cleared its slot
        assert mp.playback_holds == 0

    async def test_a_failed_resolve_leaves_a_shared_join_running(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The creator's resolve fails while a participant waits on the join it
        created: the join is the participant's now, and so is the teardown."""
        join_gate = asyncio.Event()
        fail_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)

        async def _resolve(_ctx: Any, _source: Any, **kw: Any) -> QueueObject:
            if kw["origin"] == "bad":
                await fail_gate.wait()
                raise RuntimeError("no such song")
            return song(2, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_resolve)
        music_bot._command_error = AsyncMock()
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            creator = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="bad")
            )
            await settle()
            participant = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url="good")
            )
            await settle()
            assert mp.playback_holds == 2
            fail_gate.set()
            await creator
            join = music_bot._plays._guilds[play_key(mock_ctx)].join
            assert join is not None and not join.cancelling()
            join_gate.set()
            await participant

        music_bot.cleanup.assert_not_awaited()
        mp.queue_put_front.assert_awaited_once()
        assert mp.playback_holds == 0

    async def test_a_join_being_cancelled_is_not_handed_to_the_next_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A creator that fails alone cancels its join, which settles a tick
        later. A request arriving in that tick gets a fresh join, not one about
        to raise at it."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        outcomes = iter([RuntimeError("no such song"), song(2, mock_ctx)])

        async def _resolve(*_a: Any, **_k: Any) -> QueueObject:
            outcome = next(outcomes)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        play_pipeline.queue_source = AsyncMock(side_effect=_resolve)
        music_bot._command_error = AsyncMock()
        music_bot.cleanup = AsyncMock()

        with no_typing("src.commands.play.background_typing"):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await settle()
            # call_count: the dying join never ran, so it was never awaited.
            assert mock_ctx.invoke.call_count == 2  # a second join, not the dying one
            join_gate.set()
            await asyncio.gather(*tasks)

        mp.queue_put_front.assert_awaited_once()
        assert mp.playback_holds == 0

    async def test_a_failed_join_is_torn_down_exactly_once(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Last one out: the first failure sees the other's hold and skips, the
        second sees none and tears down."""
        join_gate = asyncio.Event()
        mp = self._cold(music_bot, mock_ctx, join_gate=join_gate)
        songs = iter([song(1, mock_ctx), song(2, mock_ctx)])
        play_pipeline.queue_source = AsyncMock(
            side_effect=lambda *_a, **_k: next(songs)
        )
        music_bot.cleanup = AsyncMock()

        async def _failed_join(*_a: Any, **_k: Any) -> None:
            await join_gate.wait()  # voice_client stays None: no usable client

        mock_ctx.invoke = AsyncMock(side_effect=_failed_join)

        with no_typing("src.commands.play.background_typing"):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(music_bot, mock_ctx, url=f"s{n}")
                )
                for n in (1, 2)
            ]
            await settle()
            join_gate.set()
            await asyncio.gather(*tasks)

        music_bot.cleanup.assert_awaited_once()
        mp.queue_put_front.assert_not_awaited()

    def test_teardown_decision_and_hold_release_have_no_await_between_them(
        self,
    ) -> None:
        """_abandon_cold_start's skip reads the other participant's hold, and the
        hold is released by leaving the stack. An await between the two — a
        notice, a log flush — lets both participants skip."""
        import inspect

        lines = inspect.getsource(play_cmd._resolve_and_place).splitlines()
        decisions = [
            i for i, line in enumerate(lines) if "await abandon_cold_start(" in line
        ]
        assert decisions, "the cold-start block moved; re-anchor this test"
        for i in decisions:
            following = next(
                line.strip()
                for line in lines[i + 1 :]
                if line.strip() and not line.strip().startswith("#")
            )
            assert following.startswith(("return", "raise")), (i, following)
        # And the decision itself: _abandon_cold_start reads the OTHER
        # participant's hold and returns on it. An await between those two lets
        # that participant reach its own decision while this hold is still
        # counted, and then both skip and nobody tears the player down.
        guard = inspect.getsource(abandon_cold_start).splitlines()
        read = next(n for n, line in enumerate(guard) if "playback_holds > 1" in line)
        assert guard[read + 1].strip().startswith("return"), guard[read + 1]


class TestPlacementRevalidationCarriesDispatch:
    """place() re-runs the voice check after the resolve, on the dispatch-time
    reading of what this -play is."""

    async def test_a_pause_during_the_resolve_does_not_refuse_a_plain_play(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A plain -play from another channel is exempt from the same-channel rule —
        appending costs the listeners there nothing. Re-deriving queue_control under
        the lock reads a pause that landed during the resolve and refuses an append
        that takes nothing from anyone."""
        vc = playing_vc(mock_ctx)
        vc.is_paused = MagicMock(return_value=False)
        vc.channel = MagicMock()  # the author is somewhere else
        mock_ctx.voice_client = vc
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)

        async def _pause_mid_resolve(*_a: Any, **_kw: Any) -> QueueObject:
            vc.is_paused.return_value = True
            return song(1, mock_ctx)

        play_pipeline.queue_source = AsyncMock(side_effect=_pause_mid_resolve)
        with no_typing("src.commands.play.background_typing"):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="a")

        mp.queue_put.assert_awaited()

    async def test_a_now_is_still_gated_on_the_same_channel(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The carried value must not become a blanket exemption: --now is queue
        control at dispatch and stays gated."""
        vc = playing_vc(mock_ctx)
        vc.is_paused = MagicMock(return_value=False)
        vc.channel = MagicMock()
        mock_ctx.voice_client = vc
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        play_pipeline.queue_source = AsyncMock(return_value=song(1, mock_ctx))

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock()),
        ):
            await command_callback(MusicBot.play)(
                music_bot, mock_ctx, url="--next test"
            )

        mp.queue_put_next.assert_not_awaited()


class TestTheInterjectionHeadMustBePlayable:
    """This flow stops what is already playing, so the head has to be known
    playable first — and a ytdl:source entry written by an earlier flat resolve
    carries no stream URL, so a FULL resolve can reach here having proved nothing."""

    @pytest.fixture
    def live_mp(self) -> MagicMock:
        from src.musicplayer import InterjectOutcome

        mp = mock_mp()
        mp.current_song = MagicMock()
        mp.interject = AsyncMock(
            return_value=InterjectOutcome(
                interrupted_title="Original Song",
                resume_position=151,
                was_paused=False,
                returns_paused=False,
            )
        )
        return mp

    @pytest.fixture
    def live_vc(self, mock_ctx: MagicMock) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return in_authors_channel(vc, mock_ctx)

    async def test_an_unwarmable_head_does_not_stop_the_song(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock(return_value=False)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        live_mp.interject.assert_not_awaited()
        live_vc.stop.assert_not_called()
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "left alone" in text

    async def test_a_warmed_head_interjects(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The gate must not refuse the ordinary case."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        play_pipeline.queue_source = AsyncMock(
            return_value=QueueObject("https://yt.com/v=x", "Urgent", mock_ctx.author)
        )

        with (
            no_typing("src.commands.play.background_typing"),
            patch.object(YTDL, "prefetch_stream", new=AsyncMock(return_value=True)),
        ):
            await command_callback(MusicBot.play)(music_bot, mock_ctx, url="--now test")

        live_mp.interject.assert_awaited()


class TestAQuotedRequestIsDroppable:
    """-remove matches the origin the queue entry carries, which _play unquotes."""

    async def test_a_quoted_in_flight_request_carries_the_unquoted_query(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        mock_ctx.voice_client = playing_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        gate = asyncio.Event()
        play_pipeline.queue_source = _gated_resolve(song(1, mock_ctx), gate)

        with no_typing("src.commands.play.background_typing"):
            task = asyncio.create_task(
                command_callback(MusicBot.play)(music_bot, mock_ctx, url='"some song"')
            )
            await settle()
            inflight = music_bot._plays._guilds[play_key(mock_ctx)].inflight
            assert [r.query for r in inflight] == ["some song"]
            gate.set()
            await task
