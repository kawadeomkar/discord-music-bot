"""Tests for `-replay` (src/commands/replay.py)."""

import asyncio
import contextlib
from collections.abc import Generator, Iterator
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.commands._common import NOTHING_PLAYING
from src.guild_state import Analytics
from src.commands import replay as replay_cmd
from src.commands.replay import ReplayOutcome, ReplayResult
from src.guild_queue import GuildQueue
from src.guild_state import SongQueueEntry, parse_queue_entry
from src.musicbot import MusicBot
from src.musicplayer import MusicPlayer
from src.util import cancel_task
from src.youtube import YTDL, QueueObject
from tests.helpers import (
    REPLAY_ASK,
    command_callback,
    queue_object,
    replayed_song,
)


class TestReplayCommand:
    @pytest.fixture
    def live_vc(self) -> MagicMock:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        return vc

    @pytest.fixture
    def live_mp(self) -> MagicMock:
        """A MusicPlayer mock with a song playing, replayed successfully."""
        mp = MagicMock()
        # A real position: the command refuses a song still at its beginning, and a
        # bare MagicMock's __int__ answers 1 — right at the threshold, so the guard
        # would be passing by accident rather than by intent.
        mp.current_song = MagicMock()
        mp.current_song.position_secs = 151.0
        mp.current_song.title = "Original Song"
        # A bare MagicMock answers truthy, which reads as a claimed next song.
        mp.queue.claim_outstanding = MagicMock(return_value=False)
        return mp

    @pytest.fixture(autouse=True)
    def replay(self) -> Iterator[AsyncMock]:
        """replay_current, stubbed to a successful replay: these tests are the
        command's refusals and wording. TestReplayCurrent drives the real flow
        against a MusicPlayer."""
        stub = AsyncMock(
            return_value=ReplayOutcome(title="Original Song", position=151)
        )
        with patch("src.commands.replay.replay_current", new=stub):
            yield stub

    async def test_replays_the_live_song(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        replay.assert_awaited_once()
        assert (call := replay.await_args) is not None
        assert call.args == (live_mp, live_vc)
        # The ask is this message by this caller, and it plays immediately. Both
        # columns must name the same person: split, the archive row claims the
        # original requester asked for the song at the moment someone else typed.
        assert call.kwargs["analytics"] == Analytics(
            queued_at=mock_ctx.message.created_at.timestamp(), queue_position=0
        )
        assert call.kwargs["requester"] is mock_ctx.author
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Original Song" in embed.description
        assert "2:31" in embed.description
        # Both wordings name the title and the position, so only the state
        # separates them — asserting the two alone passes for either.
        assert "from `0:00`" in embed.description
        assert "paused" not in embed.description
        assert embed.color == discord.Color.blue()
        mock_ctx.message.add_reaction.assert_awaited_once_with("🔁")
        # A replay that happened spends the cooldown: it writes history.
        mock_ctx.command.reset_cooldown.assert_not_called()

    async def test_a_paused_song_is_replayed_and_confirmed_the_same_way(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        """A paused song comes back playing, so there is one wording rather than
        two. The dispatch guard admits a paused voice client for the same reason:
        refusing there would make -replay the one playback verb a pause turns
        off."""
        live_vc.is_playing.return_value = False
        live_vc.is_paused.return_value = True
        replay.return_value = ReplayOutcome(title="Original Song", position=151)
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        replay.assert_awaited_once()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed.color == discord.Color.blue()
        assert "from `0:00`" in embed.description
        assert "paused" not in embed.description

    @pytest.mark.parametrize(
        "state",
        [
            pytest.param({"current_song": None}, id="nothing-live"),
            pytest.param({"voice_client": None}, id="not-in-voice"),
            pytest.param({"idle": True}, id="connected-but-idle"),
        ],
    )
    async def test_reports_nothing_playing(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
        state: dict[str, Any],
    ) -> None:
        if "current_song" in state:
            live_mp.current_song = None
        if state.get("idle"):
            live_vc.is_playing.return_value = False
            live_vc.is_paused.return_value = False
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc if "voice_client" not in state else None

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        replay.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed.description == "No songs are currently playing."
        assert embed.color == discord.Color.orange()
        mock_ctx.message.add_reaction.assert_not_awaited()
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

    async def test_song_ending_mid_replay_is_not_reported_as_idle(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        """replay_current returns None when the song ends inside its stream warm.
        Something WAS playing at dispatch, so the generic idle notice would read as
        the bot having ignored the command."""
        replay.return_value = None
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "no longer playing" in embed.description
        mock_ctx.message.add_reaction.assert_not_awaited()
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

    async def test_refuses_a_song_still_at_its_beginning(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        """Nothing to rewind — and it is also how a repeat mints history entries
        nobody heard: a replay parks at 0:00, so replaying it again stops a song
        with no frames, which the loop still records. Each such row LTRIMs a real
        play out of the 50-entry window, and with the archive off that list is the
        only record there is."""
        live_mp.current_song.position_secs = 0.4
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        replay.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "already at the beginning" in embed.description
        assert "Original Song" in embed.description
        mock_ctx.message.add_reaction.assert_not_awaited()
        # Answered two seconds later with "on cooldown" otherwise: the retry the
        # refusal all but invites is the one the cooldown then blocks.
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

    async def test_a_song_that_ended_first_is_reported_as_queued_not_replayed(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        """The stop is declined when the loop moved on while the replay resolved.
        The replay is real — it is at the queue front and plays next — but nothing
        was interrupted, so "Replaying … was at 2:31" describes an event that did
        not happen, on a song the user watched end."""
        replay.return_value = ReplayOutcome(
            title="Original Song", position=151, result=ReplayResult.ENDED_FIRST
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert "ended first" in description
        assert "playing next" in description
        assert "was at" not in description

    @pytest.mark.parametrize(
        "result,said,reacts",
        [
            pytest.param(
                ReplayResult.STILL_LOADING, "once this play ends", True, id="loading"
            ),
            pytest.param(ReplayResult.FAILED, "keeps playing", False, id="failed"),
            pytest.param(
                ReplayResult.DROPPED, "taken out of the queue", False, id="dropped"
            ),
            pytest.param(
                ReplayResult.QUEUED_LATER, "once the queue reaches it", True, id="later"
            ),
            pytest.param(
                ReplayResult.INTERRUPTED, "not replayed now", True, id="interrupted"
            ),
            pytest.param(
                ReplayResult.STOPPED_ELSEWHERE,
                "Another command stopped",
                True,
                id="stopped-elsewhere",
            ),
            pytest.param(ReplayResult.TORN_DOWN, "`-resume`", True, id="torn-down"),
        ],
    )
    async def test_a_replay_that_stopped_nothing_says_why(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
        result: ReplayResult,
        said: str,
        reacts: bool,
    ) -> None:
        """The song is still playing in all three, so "Replaying … was at 2:31"
        would describe a stop that did not happen. Only a copy that will still play
        earns the 🔁."""
        replay.return_value = ReplayOutcome(
            title="Original Song", position=151, result=result
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert said in description
        assert "Original Song" in description
        assert "Replaying" not in description
        assert mock_ctx.message.add_reaction.await_count == (1 if reacts else 0)
        # A copy that still plays writes history; one that does not, does not.
        assert mock_ctx.command.reset_cooldown.call_count == (0 if reacts else 1)

    @pytest.mark.parametrize(
        "setup,outcome",
        [
            pytest.param("replayed", "replaying", id="replayed"),
            pytest.param("at_beginning", "at_beginning", id="at-beginning"),
            pytest.param("idle", "idle", id="idle"),
            pytest.param("not_live", "not_live", id="not-live"),
        ],
    )
    async def test_the_command_span_names_how_it_ended(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
        setup: str,
        outcome: str,
    ) -> None:
        """player.replay records only the replays that reached the player; a refusal
        left a bare bot.replay span."""
        if setup == "at_beginning":
            live_mp.current_song.position_secs = 0.2
        elif setup == "idle":
            live_mp.current_song = None
            live_mp.queue.empty = MagicMock(return_value=True)
        elif setup == "not_live":
            replay.return_value = None
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        # The body, not the cog's callback: its decorator opens bot.replay on the
        # module's own tracer, which records nothing here.
        with provider.get_tracer("test").start_as_current_span("bot.replay"):
            await replay_cmd.run(mock_ctx, mp=live_mp)

        (span,) = exporter.get_finished_spans()
        assert (span.attributes or {})["replay.outcome"] == outcome

    async def test_the_reaction_does_not_wait_for_the_confirmation(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        reacted = asyncio.Event()

        async def _slow_send(**_: Any) -> None:
            async with asyncio.timeout(2):
                await reacted.wait()

        mock_ctx.send = AsyncMock(side_effect=_slow_send)
        mock_ctx.message.add_reaction = AsyncMock(side_effect=lambda _e: reacted.set())
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        assert reacted.is_set()
        mock_ctx.send.assert_awaited_once()

    async def test_a_failed_confirmation_still_reaches_the_error_embed(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """Only the reaction's failure is swallowed: gathered, a failed send must
        still render as a failure rather than vanish."""
        mock_ctx.send = AsyncMock(
            side_effect=[discord.HTTPException(MagicMock(status=500), "down"), None]
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        assert mock_ctx.send.await_args.kwargs["embed"].title == "Failed to replay song"

    async def test_a_missing_reaction_permission_does_not_report_failure(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """gather does not cancel siblings, so the confirmation lands and the raise
        then reaches the command's except — rendering a red "Failed to replay
        song" beside it for a replay that has already happened and cannot be
        undone. The reaction is decoration; the send is the answer."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        mock_ctx.message.add_reaction = AsyncMock(
            side_effect=discord.HTTPException(MagicMock(), "missing permissions")
        )

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        assert mock_ctx.send.await_count == 1
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Replaying" in embed.description
        assert embed.color == discord.Color.blue()

    async def test_the_gap_between_songs_is_not_reported_as_an_idle_guild(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        """current_song is None for the 1-4s a connected bot spends resolving the
        next song. Answering that with the idle notice tells a user watching a
        queue they can see that there is nothing playing."""
        live_mp.current_song = None
        live_mp.queue.empty = MagicMock(return_value=False)
        live_vc.is_playing.return_value = False
        live_vc.is_paused.return_value = False
        live_vc.is_connected.return_value = True
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert "still loading" in description
        replay.assert_not_awaited()
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

    @pytest.mark.parametrize(
        "claimed,pending,connected,loading",
        [
            pytest.param(True, False, True, True, id="last-song-claimed"),
            pytest.param(False, True, True, True, id="songs-pending"),
            pytest.param(False, True, False, False, id="disconnected-with-a-queue"),
        ],
    )
    async def test_between_songs_is_told_apart_from_idle(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
        claimed: bool,
        pending: bool,
        connected: bool,
        loading: bool,
    ) -> None:
        """With one song left, the prefetch holds it as a claim and the queue
        reads empty: "nothing playing" a moment before it starts. Nothing invites
        a retry, which would replay whatever starts next."""
        live_mp.current_song = None
        live_mp.queue.claim_outstanding = MagicMock(return_value=claimed)
        live_mp.queue.empty = MagicMock(return_value=not pending)
        live_vc.is_playing.return_value = False
        live_vc.is_paused.return_value = False
        live_vc.is_connected.return_value = connected
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert ("still loading" in description) is loading
        assert "again" not in description

    @pytest.mark.parametrize("position,refused", [(0.99, True), (1.0, False)])
    async def test_the_beginning_ends_at_one_second(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
        position: float,
        refused: bool,
    ) -> None:
        live_mp.current_song.position_secs = position
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        assert replay.await_count == (0 if refused else 1)

    async def test_an_idle_guild_with_no_queue_still_gets_the_shared_notice(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """The other side of the branch above: -now and -replay answer the same
        state, so they must answer it with the same sentence."""
        live_mp.current_song = None
        live_mp.queue.empty = MagicMock(return_value=True)
        live_vc.is_playing.return_value = False
        live_vc.is_paused.return_value = False
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed.description == NOTHING_PLAYING

    async def test_shows_typing_while_the_replay_resolves(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        """replay_current awaits a yt-dlp resolve on a cold cache. Without the
        wrapper the user gets no ack, no reaction and no typing for seconds, and a
        second -replay in that window is declined by max_concurrency — so the bot
        looks like it ignored them. Every sibling that awaits an extraction
        (-play, -playnow, -resume, -shuffle) wraps its body the same way."""
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc
        typing_cm = MagicMock(return_value=contextlib.nullcontext())

        with patch("src.commands.replay.background_typing", typing_cm):
            await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        typing_cm.assert_called_once_with(mock_ctx)

    async def test_failure_renders_the_command_error(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        replay: AsyncMock,
        live_vc: MagicMock,
    ) -> None:
        replay.side_effect = RuntimeError("boom")
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert embed.title == "Failed to replay song"
        assert embed.color == discord.Color.red()


class TestReplayDecorators:
    """command_callback() hands back the raw callback, so every test above runs with
    the decorators bypassed. Nothing else reads them."""

    def test_replay_is_rate_limited_as_well_as_serialized(self) -> None:
        """max_concurrency bounds how many run at once; it does not bound how OFTEN.
        -replay is the one command that consumes nothing and can be repeated on the
        same song forever, and every repeat writes a history entry — which LTRIMs a
        real play out of the 50-entry window that, with the archive off, is the only
        record a guild has. Deleting the decorator left the whole suite green."""
        buckets = MusicBot.replay._buckets
        assert buckets.valid, "-replay lost its cooldown"
        assert buckets._cooldown is not None
        assert buckets._cooldown.rate == 1
        assert buckets._cooldown.per == 5.0
        assert buckets.type is commands.BucketType.guild

    def test_replay_advertises_only_aliases_it_answers_to(self) -> None:
        """-help prints these as runnable examples. A typo in the tuple — `restrat`
        for `restart` — survives the whole suite while every example the help embed
        prints for that alias 404s, because nothing else reads both."""
        assert set(MusicBot.replay.aliases) == {"rp", "restart"}
        examples = MusicBot.replay.extras["examples"]
        names = {MusicBot.replay.name, *MusicBot.replay.aliases}
        for example in examples:
            assert example.lstrip("-").split()[0] in names, example

    def test_replay_requires_the_author_in_the_voice_channel(self) -> None:
        """command_callback() hands back the raw callback, so every test of -replay
        runs with its decorators bypassed — deleting this one leaves the suite green
        while the command reaches ctx.voice_client for a user who is not in the
        channel, and replays a song for a guild the caller is not listening to."""
        assert MusicBot.replay._before_invoke is MusicBot.validate_commands


async def _claim_the_head(mp: MusicPlayer) -> Optional[MagicMock]:
    """What _prefetch_next_song does on success: claim the head and hand back its
    stream, leaving the claim for the loop to settle."""
    if mp.queue.empty():
        return None
    mp.queue.get_nowait()
    return MagicMock(spec=YTDL)


class TestReplayCurrent:
    @pytest.fixture(autouse=True)
    def _stub_replay_resolve(self, music_player: MusicPlayer) -> Generator[AsyncMock]:
        """replay_current resolves the replay through the loop's own prefetch path.
        Stubbed here so these tests stay at the flow; TestReplayLoopStart in
        tests/test_musicplayer.py drives the real one. The stub claims the head as the real one does: the
        stop is gated on the copy being held there."""

        async def resolve() -> Optional[MagicMock]:
            return await _claim_the_head(music_player)

        stub = AsyncMock(side_effect=resolve)
        with patch.object(MusicPlayer, "_prefetch_next_song", new=stub):
            yield stub

    async def test_returns_none_without_current_song(
        self, music_player: MusicPlayer, mock_vc: MagicMock, replayer: MagicMock
    ) -> None:
        music_player.current_song = None
        assert (
            await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
            is None
        )
        mock_vc.stop.assert_not_called()

    async def test_returns_none_without_a_url_to_rebuild_from(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """An entry built from an empty webpage_url would fail at the next resolve
        instead of replaying anything."""
        live_song.webpage_url = ""
        music_player.current_song = live_song

        assert (
            await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
            is None
        )

        assert music_player.queue.display_items() == []
        mock_vc.stop.assert_not_called()

    async def test_front_inserts_a_copy_starting_from_the_beginning(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
        replayer: MagicMock,
    ) -> None:
        live_song.elapsed_secs = 42.0
        music_player.current_song = live_song
        queued = QueueObject("https://yt.com/v=b", "Queued B", mock_author)
        await music_player.queue.put([queued])

        outcome = await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        items = music_player.queue.display_items()
        replay = items[0]
        assert isinstance(replay, QueueObject)
        assert replay.webpage_url == live_song.webpage_url
        assert replay.title == live_song.title
        assert replay.duration == live_song.duration_secs
        # Carried so the queue card and the NP block render the same song they
        # would have rendered for the original entry.
        assert replay.uploader == live_song.uploader
        assert replay.thumbnail == live_song.thumbnail
        # No -ss and no resume wording: this play starts at 0:00.
        assert replay.ts is None
        assert replay.is_resume is False
        assert replay.start_paused is False
        # Marks the queue card, so the entry does not read as the live song
        # queued behind itself.
        assert replay.is_replay is True
        assert replay.persisted is True
        assert items[1] is queued  # the queue behind it is untouched

        mock_vc.stop.assert_called_once()
        assert outcome is not None
        assert outcome.title == live_song.title
        assert outcome.position == 42
        assert outcome.position_str == "0:42"

    async def test_the_replay_is_unstamped_even_though_the_live_song_is_not(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The live song ALWAYS carries a played_at — the loop stamps it at
        vc.play(). Carrying it onto the replay would be invisible here if the
        fixture were left unstamped, and in production it deletes an archive row:
        the loop's stamp is `played_at or now`, so an inherited value survives, both
        rows share the (guild_id, played_at, webpage_url) dedup key, and ON CONFLICT
        DO NOTHING drops the second with no error and no log line."""
        live_song.played_at = 1752530000.0
        music_player.current_song = live_song

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        replay = music_player.queue.display_items()[0]
        assert isinstance(replay, QueueObject)
        assert replay.played_at == 0.0

    async def test_position_counts_the_start_offset(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """position_secs is start_offset + elapsed. Reading elapsed_secs instead
        would report `0:12` for a `?t=180` link or a --now tail parked at 3:00 —
        and this number is the only thing the reply tells the user about what they
        interrupted."""
        live_song.start_offset = 180
        live_song.elapsed_secs = 12.0
        music_player.current_song = live_song

        outcome = await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert outcome is not None
        assert outcome.position == 192
        assert outcome.position_str == "3:12"

    async def test_replay_carries_the_query_source_and_user_input(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """Neither is recoverable from webpage_url — a Spotify link, a search and a
        pasted link all archive as youtube.com — and a replay does not change where
        the song came from. Dropped here, the replay's history row reads as a
        pre-feature one and -remove can no longer take it back out by its input."""
        live_song.query_source = "spotify.com"
        live_song.user_input = "https://open.spotify.com/track/abc"
        music_player.current_song = live_song

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        replay = music_player.queue.display_items()[0]
        assert isinstance(replay, QueueObject)
        assert replay.query_source == "spotify.com"
        assert replay.user_input == "https://open.spotify.com/track/abc"

    async def test_the_replay_is_the_callers_ask_in_both_columns(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """requester and analytics.queued_at both describe who asked and when. Split
        them — the original requester with the replayer's timestamp — and the row
        claims someone asked for a song at a moment they did not, which
        -leaderboard sums into their listening time."""
        live_song.requester = mock_author
        live_song.analytics = Analytics(queued_at=1752530000.5, queue_position=5)
        music_player.current_song = live_song

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        replay = music_player.queue.display_items()[0]
        assert isinstance(replay, QueueObject)
        assert replay.requester is replayer
        assert replay.analytics == REPLAY_ASK

    async def test_a_paused_song_comes_back_playing(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """-play's precedent: the caller named THIS song, so the ask is for it to
        sound. Parked instead, the replay is a bot making no noise under a card that
        says Now Playing — which reads as the command having been ignored."""
        mock_vc.is_playing.return_value = False
        mock_vc.is_paused.return_value = True
        live_song.elapsed_secs = 90.0
        music_player.current_song = live_song

        outcome = await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        replay = music_player.queue.display_items()[0]
        assert isinstance(replay, QueueObject)
        assert replay.start_paused is False
        assert outcome is not None
        assert outcome.position == 90
        mock_vc.stop.assert_called_once()  # a paused song is stopped too

    async def test_the_replay_reaches_redis_as_a_play_from_the_start(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The mirror, not the in-memory copy, is what a crash mid-replay recovers
        from — and put_front serializes the entry inside the call, so a field set on
        the object afterwards lands in memory and `false` on the wire. Recovered
        from a wrong entry the replay resumes partway in, or paused, or both."""
        live_song.elapsed_secs = 90.0
        music_player.current_song = live_song

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert music_player.store is not None
        raw = await music_player.store.redis.lrange(
            f"guild:{music_player._guild.id}:queue", 0, -1
        )
        assert raw
        entry = parse_queue_entry(raw[0])
        assert isinstance(entry, SongQueueEntry)
        assert not entry.ts
        assert entry.is_resume is False
        assert entry.start_paused is False

    async def test_declines_a_song_another_interrupt_already_stopped(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """-replay's max_concurrency bucket is its own and --now admits through
        PlayRegistry, so neither declines the other; the loop has not woken, so
        current_song still names the song --now just stopped. Both insert, and
        the listener hears the song three times."""
        music_player.current_song = live_song
        music_player.note_deliberate_stop()

        assert (
            await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
            is None
        )
        assert music_player.queue.display_items() == []
        mock_vc.stop.assert_not_called()

    async def test_declines_a_song_skip_already_stopped(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """-skip stops the song and returns; until the audio thread reports it,
        current_song still names that song. A -replay there would queue the song
        the user just skipped."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song
        music_player.note_deliberate_stop()  # what -skip does before vc.stop()

        assert (
            await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
            is None
        )
        assert music_player.queue.display_items() == []
        mock_vc.stop.assert_not_called()
        assert music_player._retire_np_for is None

    async def test_a_skip_during_the_resolve_is_not_reported_as_a_replay(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The replay is already queued and plays next, but -skip, not this
        command, stopped the song: no second stop, and no card retired on its
        behalf."""
        live_song.elapsed_secs = 30.0
        music_player.current_song = live_song

        async def resolve_while_skipped(_self: Any) -> None:
            music_player.note_deliberate_stop()

        with patch.object(
            MusicPlayer, "_prefetch_next_song", new=resolve_while_skipped
        ):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is not None
        assert outcome.result is ReplayResult.STOPPED_ELSEWHERE
        mock_vc.stop.assert_not_called()
        assert music_player._retire_np_for is None
        assert len(music_player.queue.display_items()) == 1

    async def test_a_teardown_during_the_resolve_says_the_copy_waits_for_resume(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """A -stop, a kick or the alone watchdog leaves the copy in the saved
        queue: "ended first — playing next" would describe a bot that has left."""
        music_player.current_song = live_song
        finished = asyncio.create_task(asyncio.sleep(0))
        await finished

        async def resolve_while_torn_down(_self: Any) -> None:
            music_player.note_deliberate_stop()  # cleanup stops the song too
            music_player._player = finished

        with patch.object(
            MusicPlayer, "_prefetch_next_song", new=resolve_while_torn_down
        ):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is not None
        assert outcome.result is ReplayResult.TORN_DOWN
        mock_vc.stop.assert_not_called()

    async def test_a_now_during_the_resolve_plays_the_song_twice_not_three_times(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
        replayer: MagicMock,
        _stub_replay_resolve: AsyncMock,
    ) -> None:
        """--now neutralizes the replay's resolve, which hands the copy back to the
        head, then front-inserts. A resume tail there would sit ahead of the copy:
        the song, the interjection, the rest of the song, then all of it again."""
        live_song.elapsed_secs = 83.0
        music_player.current_song = live_song
        claimed = asyncio.Event()

        async def resolve_forever() -> None:
            # Gives its claim back on cancel, as _prefetch_next_song does.
            item = music_player.queue.get_nowait()
            claimed.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                music_player.queue.requeue_front(item)
                raise

        _stub_replay_resolve.side_effect = resolve_forever
        replaying = asyncio.create_task(
            replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
        )
        async with asyncio.timeout(5):
            await claimed.wait()
        interjection = QueueObject("https://yt.com/v=x", "Song X", mock_author)

        outcome = await music_player.interject(interjection, mock_vc)
        async with asyncio.timeout(5):
            replayed = await replaying

        assert outcome is not None and outcome.replay_pending
        assert outcome.resume_position is None
        queued = [queue_object(item) for item in music_player.queue.display_items()]
        assert [q.title for q in queued] == ["Song X", live_song.title]
        assert queued[1].is_replay and not queued[1].is_resume
        # No tail, so the interrupted fragment records its own row, as with -skip.
        assert music_player._skip_history_for is None
        assert replayed is not None
        assert replayed.result is ReplayResult.INTERRUPTED
        mock_vc.stop.assert_called_once()  # --now's stop, not a second one

    async def test_declines_once_the_player_has_been_torn_down(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """cleanup() cancels the loop task but never clears current_song, so a -stop
        (or the alone watchdog, or a voice kick) completing here would otherwise
        stop a disconnected voice client and tell the user a bot that has already
        left is replaying something."""
        finished = asyncio.create_task(asyncio.sleep(0))
        await finished
        music_player._player = finished
        music_player.current_song = live_song

        assert (
            await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
            is None
        )
        mock_vc.stop.assert_not_called()

    async def test_a_resolve_cancelled_by_a_teardown_does_not_raise(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """cleanup() cancels _prefetch_task, which is the very task this awaits.
        Awaited directly it re-raises CancelledError into the command body, where
        `except Exception` does not catch it — asyncio.wait reports instead. A
        cancelled resolve holds no copy, so nothing is stopped for it; the copy is
        still next and resolves at its dequeue."""

        async def cancel_self(_self: Any) -> None:
            task = asyncio.current_task()
            assert task is not None
            task.cancel()
            await asyncio.sleep(0)

        music_player.current_song = live_song
        with patch.object(MusicPlayer, "_prefetch_next_song", new=cancel_self):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is not None
        assert outcome.result is ReplayResult.INTERRUPTED
        mock_vc.stop.assert_not_called()

    async def test_leaves_the_interrupted_play_to_record_itself(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The opposite of a --now interjection, deliberately: a resume tail
        SPANS what was already heard, so the fragment declines its entry — a replay
        starts at 0:00 and spans nothing, so suppressing the interrupted play would
        lose that listening outright."""
        live_song.elapsed_secs = 42.0
        live_song.produced_audio = True
        music_player.current_song = live_song

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert music_player._skip_history_for is None

    async def test_marks_the_interrupted_card_for_retirement(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """Left to finalize, the interrupted bar stays in the channel frozen at its
        stop position — directly above the replay's own card, naming the same song.
        Two identical Now Playing cards read as a duplicate queue entry, not as
        history."""
        music_player.current_song = live_song

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert music_player._retire_np_for is live_song
        # Same stop, recorded for the other axis: a --now landing before the loop
        # wakes sees a song already stopped and bails to its own fallback.
        assert music_player._stopped_deliberately

    async def test_resolves_the_replay_before_stopping_the_song(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
        _stub_replay_resolve: AsyncMock,
    ) -> None:
        """The extraction, the probe and the FFmpeg spawn are paid while the song is
        still playing, not in the silence after the stop. The task is left on
        _prefetch_task so the loop consumes it as an ordinary prefetch."""
        music_player.current_song = live_song
        order: list[str] = []
        mock_vc.stop = MagicMock(side_effect=lambda: order.append("stop"))

        async def resolve_slowly() -> Optional[MagicMock]:
            # Slower than a tick and well inside the real bound: a bound of zero,
            # or a stop that did not wait, stops before this finishes.
            await asyncio.sleep(0.05)
            order.append("resolve")
            return await _claim_the_head(music_player)

        _stub_replay_resolve.side_effect = resolve_slowly

        outcome = await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert order == ["resolve", "stop"]
        assert outcome is not None and outcome.result is ReplayResult.REPLAYING
        assert music_player._prefetch_task is not None

    async def test_a_slow_resolve_does_not_hold_the_command(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
        _stub_replay_resolve: AsyncMock,
    ) -> None:
        """A cold extraction has no upper bound of its own (the pool sets no
        timeout and yt-dlp retries), so the wait is bounded here. Expiring stops
        nothing: stopped into a resolve that might still fail, the song would be
        cut off for a copy that never plays. The copy stays held and follows it."""
        music_player.current_song = live_song
        started = asyncio.Event()

        async def never_finishes() -> None:
            await _claim_the_head(music_player)
            started.set()
            await asyncio.sleep(3600)

        _stub_replay_resolve.side_effect = never_finishes

        with patch("src.commands.replay._REPLAY_RESOLVE_TIMEOUT", 0.05):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is not None
        assert outcome.result is ReplayResult.STILL_LOADING
        assert started.is_set()
        mock_vc.stop.assert_not_called()
        assert music_player._retire_np_for is None
        # Shielded, so the timeout did not cancel the work the loop will consume.
        assert music_player._prefetch_task is not None
        assert not music_player._prefetch_task.done()
        await cancel_task(music_player._prefetch_task)

    async def test_a_failed_resolve_leaves_the_song_playing(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
        _stub_replay_resolve: AsyncMock,
    ) -> None:
        """The prefetch retires a copy it could not resolve. Stopped anyway, the
        channel reads "Replaying", the song is cut off, and the next song starts —
        or, with nothing queued, silence until the idle disconnect."""
        music_player.current_song = live_song

        async def fail() -> None:
            item = music_player.queue.get_nowait()
            await music_player.queue.finish_failed_dequeue(item, context="test")

        _stub_replay_resolve.side_effect = fail

        outcome = await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert outcome is not None
        assert outcome.result is ReplayResult.FAILED
        assert outcome.stopped is False
        mock_vc.stop.assert_not_called()
        assert music_player._retire_np_for is None
        assert music_player._stopped_deliberately is False

    # What each command leaves the copy as: dropped, or still queued behind
    # something else (the shuffle is fixed to a reversal, so the copy moves).
    _LANDED = {
        "clear": ReplayResult.DROPPED,
        "shuffle": ReplayResult.QUEUED_LATER,
        "next": ReplayResult.QUEUED_LATER,
    }

    @staticmethod
    async def _land(mp: MusicPlayer, command: str, author: MagicMock) -> None:
        if command == "clear":
            await mp.queue_clear()
        elif command == "shuffle":
            with patch("random.shuffle", side_effect=lambda items: items.reverse()):
                await mp.queue_shuffle()
        else:
            await mp.queue_put_next(
                QueueObject("https://yt.com/v=n", "Next", author), prefetch=False
            )

    @pytest.mark.parametrize("command", ["clear", "shuffle", "next"])
    async def test_a_command_landing_during_the_resolve_stops_nothing(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        mock_author: MagicMock,
        replayer: MagicMock,
        _stub_replay_resolve: AsyncMock,
        command: str,
    ) -> None:
        """-clear cancels the resolve and drops the copy; -shuffle and --next cancel
        it and hand the copy back to be reordered or queued behind. The verdict runs
        as the cancel lands, before any of them has changed the queue, so it can
        only say the resolve was interrupted — and stop nothing."""
        music_player.current_song = live_song
        await music_player.queue.put(
            [
                QueueObject(f"https://yt.com/v={v}", f"Song {v}", mock_author)
                for v in "bcde"
            ]
        )
        claimed = asyncio.Event()

        async def resolve_forever() -> None:
            # Gives its claim back on cancel, as _prefetch_next_song does.
            item = music_player.queue.get_nowait()
            claimed.set()
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                music_player.queue.requeue_front(item)
                raise

        _stub_replay_resolve.side_effect = resolve_forever

        replaying = asyncio.create_task(
            replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )
        )
        async with asyncio.timeout(5):
            await claimed.wait()
        await self._land(music_player, command, mock_author)
        async with asyncio.timeout(5):
            outcome = await replaying

        assert outcome is not None
        assert outcome.result is ReplayResult.INTERRUPTED
        mock_vc.stop.assert_not_called()
        await cancel_task(music_player._prefetch_task)

    @pytest.mark.parametrize("command", ["clear", "shuffle", "next"])
    async def test_a_command_landing_after_the_resolve_stops_nothing(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_author: MagicMock,
        command: str,
    ) -> None:
        """The same commands against a COMPLETED resolve, in the tick between it
        finishing and the verdict: -clear's cancel is a no-op on a done task and
        empties the deque around the claim, and a neutralize requeues a REBUILT
        copy, so identity with the copy is what tells them apart."""
        music_player.current_song = live_song
        replay = QueueObject(
            live_song.webpage_url, "Song A", mock_author, is_replay=True
        )
        await music_player.queue.put(
            [replay]
            + [
                QueueObject(f"https://yt.com/v={v}", f"Song {v}", mock_author)
                for v in "bcde"
            ]
        )
        music_player.queue.get_nowait()
        resolved = replayed_song(replay)
        resolving: asyncio.Task[Optional[YTDL]] = asyncio.create_task(
            asyncio.sleep(0, result=cast(YTDL, resolved))
        )
        await resolving
        music_player._prefetch_task = resolving
        assert (
            replay_cmd._replay_result(music_player, live_song, replay, resolving)
            is ReplayResult.REPLAYING
        )

        await self._land(music_player, command, mock_author)

        assert (
            replay_cmd._replay_result(music_player, live_song, replay, resolving)
            is self._LANDED[command]
        )
        await cancel_task(music_player._prefetch_task)

    async def test_marks_the_stop_as_deliberate_before_stopping(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """Unmarked, a stop inside ffmpeg's startup window looks exactly like a
        stream that never opened, and the cached URL is dropped for it."""
        music_player.current_song = live_song
        order: list[str] = []
        mock_vc.stop = MagicMock(side_effect=lambda: order.append("stop"))

        with patch.object(
            MusicPlayer,
            "note_deliberate_stop",
            side_effect=lambda: order.append("mark"),
        ):
            await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert order == ["mark", "stop"]

    async def test_neutralizes_a_running_prefetch_first(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """A completed prefetch bypasses the queue, so it would play INSTEAD of the
        front-inserted replay."""
        music_player.current_song = live_song
        blocker = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = blocker

        await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert blocker.cancelled()
        replay = music_player.queue.display_items()[0]
        assert isinstance(replay, QueueObject)
        assert replay.title == live_song.title

    async def test_a_bail_at_dispatch_leaves_the_next_songs_prefetch_alone(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The neutralize is destructive and the liveness answer does not depend on
        it, so checking only afterwards spends the next song's fully-resolved source
        to reach a refusal — the user is told nothing was replayed AND the next
        transition pays a cold extraction it had already paid for."""
        music_player.current_song = live_song
        music_player.note_deliberate_stop()  # already stopped by --now
        prefetch = asyncio.create_task(asyncio.sleep(30))
        music_player._prefetch_task = prefetch

        try:
            assert (
                await replay_cmd.replay_current(
                    music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
                )
                is None
            )
            assert not prefetch.done()
            assert music_player._prefetch_task is prefetch
        finally:
            prefetch.cancel()

    async def test_reports_the_stop_it_declined_to_make(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The song ended while the replay resolved: the replay is real and plays
        next, but nothing was interrupted. Reported as a replay, the reply names a
        position the user watched the song run past."""
        music_player.current_song = live_song

        async def resolve_and_advance(_self: Any) -> None:
            music_player.current_song = MagicMock()

        with patch.object(MusicPlayer, "_prefetch_next_song", new=resolve_and_advance):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is not None
        assert outcome.stopped is False
        mock_vc.stop.assert_not_called()
        # The replay is still queued — it plays next, which is what the reply says.
        assert len(music_player.queue.display_items()) == 1

    async def test_song_changed_during_neutralize_returns_none(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """Neutralize can block on an in-flight prefetch — if the loop moved on in
        that window, replaying the finished song would interrupt a song nobody asked
        to replay."""
        music_player.current_song = live_song

        async def neutralize_and_advance(_self: Any) -> None:
            music_player.current_song = MagicMock()

        with patch.object(
            MusicPlayer, "_neutralize_prefetch", new=neutralize_and_advance
        ):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is None
        assert music_player.queue.display_items() == []  # nothing inserted
        mock_vc.stop.assert_not_called()

    async def test_returns_none_once_the_loop_has_finished_the_song(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """current_song is NOT cleared when a song ends — the loop clears it two
        task cancels later. In that window an identity check alone passes for a song
        already finished, and the replay then plays a second full time ahead of the
        queue and earns a second full-length history row. play_next is set by the
        audio thread and cleared at the top of the next iteration, so it marks
        exactly that window."""
        music_player.current_song = live_song
        music_player.play_next.set()

        outcome = await replay_cmd.replay_current(
            music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert outcome is None
        assert music_player.queue.display_items() == []
        mock_vc.stop.assert_not_called()

    async def test_does_not_stop_a_song_that_started_during_the_insert(
        self,
        music_player: MusicPlayer,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """The replay is already at the front, so it plays next either way —
        stopping here would kill the song that just started instead."""
        music_player.current_song = live_song

        async def advance_mid_await(items: Any) -> None:
            music_player.current_song = MagicMock()

        with patch.object(GuildQueue, "put_front", side_effect=advance_mid_await):
            outcome = await replay_cmd.replay_current(
                music_player, mock_vc, requester=replayer, analytics=REPLAY_ASK
            )

        assert outcome is not None
        assert outcome.stopped is False
        mock_vc.stop.assert_not_called()
        assert music_player._retire_np_for is None
        # The loop has read this slot already; it dequeues the copy on its own.
        assert music_player._prefetch_task is None

    async def test_replays_without_redis(
        self,
        mock_bot: MagicMock,
        mock_guild: MagicMock,
        mock_channel: MagicMock,
        mock_ctx: MagicMock,
        live_song: MagicMock,
        mock_vc: MagicMock,
        replayer: MagicMock,
    ) -> None:
        """Golden rule 5: the in-memory bot keeps working. Reaching through
        self.store unguarded raises AttributeError here, which the command swallows
        into "Failed to replay song" — so -replay would never work at all on a
        Redis-less deployment and nothing would say why."""
        mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=None)
        mp._restore_complete.set()
        mp._playback_gate.set()
        mp.current_song = live_song

        outcome = await replay_cmd.replay_current(
            mp, mock_vc, requester=replayer, analytics=REPLAY_ASK
        )

        assert outcome is not None
        assert len(mp.queue.display_items()) == 1
