"""Tests for `-replay` (src/commands/replay.py)."""

import asyncio
import contextlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from discord.ext import commands

from src.commands._common import NOTHING_PLAYING
from src.guild_state import Analytics
from src.musicbot import MusicBot
from src.musicplayer import ReplayOutcome, ReplayResult
from tests.helpers import (
    command_callback,
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
        mp.replay_current = AsyncMock(
            return_value=ReplayOutcome(title="Original Song", position=151)
        )
        return mp

    async def test_replays_the_live_song(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        live_mp: MagicMock,
        live_vc: MagicMock,
    ) -> None:
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        live_mp.replay_current.assert_awaited_once()
        call = live_mp.replay_current.await_args
        assert call.args == (live_vc,)
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
        live_vc: MagicMock,
    ) -> None:
        """A paused song comes back playing, so there is one wording rather than
        two. The dispatch guard admits a paused voice client for the same reason:
        refusing there would make -replay the one playback verb a pause turns
        off."""
        live_vc.is_playing.return_value = False
        live_vc.is_paused.return_value = True
        live_mp.replay_current = AsyncMock(
            return_value=ReplayOutcome(title="Original Song", position=151)
        )
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        live_mp.replay_current.assert_awaited_once()
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

        live_mp.replay_current.assert_not_awaited()
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
        live_vc: MagicMock,
    ) -> None:
        """replay_current returns None when the song ends inside its stream warm.
        Something WAS playing at dispatch, so the generic idle notice would read as
        the bot having ignored the command."""
        live_mp.replay_current = AsyncMock(return_value=None)
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

        live_mp.replay_current.assert_not_awaited()
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
        live_vc: MagicMock,
    ) -> None:
        """The stop is declined when the loop moved on while the replay resolved.
        The replay is real — it is at the queue front and plays next — but nothing
        was interrupted, so "Replaying … was at 2:31" describes an event that did
        not happen, on a song the user watched end."""
        live_mp.replay_current = AsyncMock(
            return_value=ReplayOutcome(
                title="Original Song", position=151, result=ReplayResult.ENDED_FIRST
            )
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
        live_vc: MagicMock,
        result: ReplayResult,
        said: str,
        reacts: bool,
    ) -> None:
        """The song is still playing in all three, so "Replaying … was at 2:31"
        would describe a stop that did not happen. Only a copy that will still play
        earns the 🔁."""
        live_mp.replay_current = AsyncMock(
            return_value=ReplayOutcome(
                title="Original Song", position=151, result=result
            )
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
        live_mp.replay_current.assert_not_awaited()
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
        live_vc: MagicMock,
        position: float,
        refused: bool,
    ) -> None:
        live_mp.current_song.position_secs = position
        music_bot.get_mp = MagicMock(return_value=live_mp)
        mock_ctx.voice_client = live_vc

        await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        assert live_mp.replay_current.await_count == (0 if refused else 1)

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
        live_vc: MagicMock,
    ) -> None:
        live_mp.replay_current = AsyncMock(side_effect=RuntimeError("boom"))
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
