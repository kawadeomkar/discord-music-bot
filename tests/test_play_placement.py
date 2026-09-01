"""Tests for src/play_placement.py — the `-play` placement grammar, and the
registry whose place lock makes "check, then insert" atomic."""

import asyncio
import time
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import redis.asyncio as aioredis

import pytest
from discord.ext import commands

from src.musicbot import MusicBot
from src.play_placement import (
    PlayArgs,
    _GuildPlays,
    PlayMode,
    play_key,
    split_play_args,
)
from tests.helpers import (
    admit,
    command_callback,
    connected_vc,
    mock_mp,
    no_typing,
    recording_span,
    settle,
)


class TestSplitPlayArgs:
    """`--now`/`--next` comes off the front of -play's argument, or not at all: the
    remainder is both the search and the origin `-remove` matches on, so a flag
    stripped from mid-line would leave a value the user never typed."""

    @pytest.mark.parametrize(
        "argument,mode,query",
        [
            ("never gonna give you up", PlayMode.NORMAL, "never gonna give you up"),
            ("--now never gonna give you up", PlayMode.NOW, "never gonna give you up"),
            ("--next never gonna give", PlayMode.NEXT, "never gonna give"),
            ("--NOW song", PlayMode.NOW, "song"),
            ("--NeXt song", PlayMode.NEXT, "song"),
            ("--now   https://youtu.be/x", PlayMode.NOW, "https://youtu.be/x"),
            ("--next   https://youtu.be/x", PlayMode.NEXT, "https://youtu.be/x"),
            ("--now", PlayMode.NOW, ""),
            ("--next", PlayMode.NEXT, ""),
            ("  --now  song  ", PlayMode.NOW, "song"),
            ("  --next  song  ", PlayMode.NEXT, "song"),
            # A word that merely starts with a flag is a search, not a flag — and
            # not a typo either, on both the two-dash and one-dash side.
            ("--nowhere man", PlayMode.NORMAL, "--nowhere man"),
            ("-nowhere man", PlayMode.NORMAL, "-nowhere man"),
            ("--nextdoor", PlayMode.NORMAL, "--nextdoor"),
            ("-nextdoor", PlayMode.NORMAL, "-nextdoor"),
            # Trailing and repeated flags stay in the text: only the head is read.
            ("song --now", PlayMode.NORMAL, "song --now"),
            ("song --next", PlayMode.NORMAL, "song --next"),
            ("--now --now song", PlayMode.NOW, "--now song"),
            # The two are mutually exclusive by construction, so the second is
            # search text like any other repeat — it does not combine, and it does
            # not override.
            ("--now --next song", PlayMode.NOW, "--next song"),
            ("--next --now song", PlayMode.NEXT, "--now song"),
            ("", PlayMode.NORMAL, ""),
            ("   ", PlayMode.NORMAL, ""),
        ],
    )
    def test_the_head_decides(self, argument: str, mode: PlayMode, query: str) -> None:
        args = split_play_args(argument)
        assert (args.mode, args.query) == (mode, query)
        assert args.dash_typo is None

    @pytest.mark.parametrize(
        "argument,meant",
        [
            ("-now song", "--now"),  # one ASCII hyphen
            ("–now song", "--now"),  # en dash
            ("—now song", "--now"),  # em dash — what iOS turns a typed `--` into
            ("―now song", "--now"),  # horizontal bar
            ("-–now song", "--now"),  # mixed pair
            ("—NOW song", "--now"),  # the typo is case-insensitive too
            ("-now", "--now"),  # nothing behind it
            ("-next song", "--next"),
            ("—next song", "--next"),
            ("–NEXT song", "--next"),
            ("-–next song", "--next"),
            ("-next", "--next"),
        ],
    )
    def test_a_dash_away_from_a_flag_asks(self, argument: str, meant: str) -> None:
        """These cannot be anything but a misspelt flag, so the command asks rather
        than searching YouTube for the user's own flag — and it names the one it
        thinks was meant, which is the only reason dash_typo carries a string."""
        args = split_play_args(argument)
        assert args.dash_typo == meant
        assert args.mode is PlayMode.NORMAL

    @pytest.mark.parametrize("argument", ["--now song", "--next song"])
    def test_a_real_flag_is_never_read_as_a_typo(self, argument: str) -> None:
        """Ordering inside split_play_args is load-bearing: `--now` satisfies the
        near-miss pattern too (two dashes is within `{1,2}`), so the exact-match
        lookup has to run first or every correct invocation would be answered with
        a did-you-mean."""
        args = split_play_args(argument)
        assert args.dash_typo is None
        assert args.mode is not PlayMode.NORMAL

    @pytest.mark.parametrize(
        "argument",
        [
            "now thats what i call music",
            "now",
            "nowhere",
            "now --now",
            "next to me",
            "next",
            "nextdoor",
        ],
    )
    def test_a_bare_flag_word_is_a_search(self, argument: str) -> None:
        """The did-you-mean deliberately stops at the dash. `-p now thats what i
        call music` and `-p next to me` are real searches, and guessing there would
        break them."""
        args = split_play_args(argument)
        assert (args.mode, args.dash_typo, args.query) == (
            PlayMode.NORMAL,
            None,
            argument,
        )

    def test_the_query_keeps_its_case(self) -> None:
        """Only the head is lowercased to match the flag — the search is what the
        user typed, since it is also the origin -remove matches on."""
        assert split_play_args("--now Never Gonna GIVE").query == "Never Gonna GIVE"

    def test_it_splits_on_any_whitespace(self) -> None:
        """Discord messages carry newlines; the head is a token, not everything up
        to the first space."""
        assert split_play_args("--next\nsong") == PlayArgs(
            mode=PlayMode.NEXT, query="song"
        )

    def test_play_args_is_immutable(self) -> None:
        """Frozen: the split happens once at the top of the body and every consumer
        downstream — the gate, the branch, the origin — reads that same value."""
        args = split_play_args("--now song")
        with pytest.raises(AttributeError):
            setattr(args, "mode", PlayMode.NORMAL)


class TestPlayRegistry:
    """PlayRegistry.register / retire and the per-guild state they keep."""

    def test_register_is_synchronous_to_the_insert(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        first = admit(music_bot, mock_ctx, mp)
        second = admit(music_bot, mock_ctx, mp)

        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        # Held by identity, in arrival order: two requests for one query from one
        # author are two requests, and the drop reports read this order back.
        assert plays.inflight == [first, second]
        assert first.generation == mp.queue.generation

    def test_beyond_the_cap_the_request_is_declined(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        with patch("src.play_placement.PLAY_INFLIGHT_MAX", 2):
            admit(music_bot, mock_ctx, mp)
            admit(music_bot, mock_ctx, mp)
            with (
                recording_span() as span,
                pytest.raises(commands.MaxConcurrencyReached) as excinfo,
            ):
                admit(music_bot, mock_ctx, mp)

        # Recorded before the cap check: the declined request carries the count
        # it would have joined, and nothing else counts declines.
        span.set_attribute.assert_any_call("play.inflight", 3)
        span.set_attribute.assert_any_call("play.declined", True)
        assert excinfo.value.number == 2  # the cap, not 1: the wording keys on it
        assert len(music_bot._plays._guilds[play_key(mock_ctx)].inflight) == 2

    def test_another_guild_has_its_own_count(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        other = MagicMock()
        other.guild = MagicMock()
        other.guild.id = mock_ctx.guild.id + 1
        with patch("src.play_placement.PLAY_INFLIGHT_MAX", 1):
            admit(music_bot, mock_ctx, mp)
            admit(music_bot, other, mp)  # no raise

    def test_the_registry_is_dropped_once_idle(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        req = admit(music_bot, mock_ctx, mp)
        key = play_key(mock_ctx)
        assert key in music_bot._plays._guilds

        music_bot._plays.retire(req)
        assert not music_bot._plays._guilds

    async def test_the_registry_outlives_its_requests_while_a_join_runs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        req = admit(music_bot, mock_ctx, mp)
        key = play_key(mock_ctx)
        done = asyncio.Event()

        async def _join(*_a: Any, **_k: Any) -> None:
            await done.wait()

        mock_ctx.invoke = AsyncMock(side_effect=_join)
        music_bot._restore_tasks = set()

        join, owns_join = music_bot._plays.cold_join(
            req,
            joiner=lambda: mock_ctx.invoke(music_bot.join),
            tracked=music_bot._restore_tasks,
        )
        assert owns_join  # the first request to find no client creates it
        music_bot._plays.retire(req)
        assert key in music_bot._plays._guilds  # the join still runs

        done.set()
        await join
        await settle()
        assert not music_bot._plays._guilds

    async def test_the_cap_raise_escapes_the_command_body(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Raised in play() before _play's try/except: caught there it would
        render as "Failed to queue song"; from here it reaches cog_command_error
        and the existing decline notice."""
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot._command_error = AsyncMock()
        with patch("src.play_placement.PLAY_INFLIGHT_MAX", 1):
            admit(music_bot, mock_ctx, mp)
            with (
                no_typing("src.commands.play.background_typing"),
                pytest.raises(commands.MaxConcurrencyReached),
            ):
                await command_callback(MusicBot.play)(music_bot, mock_ctx, url="x")
        music_bot._command_error.assert_not_awaited()

    async def test_the_decline_names_the_cap_not_a_single_slot(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.command = MagicMock()
        mock_ctx.command.name = "play"
        await music_bot.cog_command_error(
            mock_ctx, commands.MaxConcurrencyReached(16, commands.BucketType.guild)
        )
        text = mock_ctx.send.await_args.kwargs["embed"].description
        assert "Too many" in text and "resolving" in text


def _extracted_song(video_id: str) -> dict[str, Any]:
    """The shape one yt-dlp stream-opts extraction returns for a link — enough for
    yt_source to build its QueueObject and warm both caches."""
    return {
        "url": f"https://r2.googlevideo.com/{video_id}?expire={int(time.time()) + 7200}",
        "webpage_url": f"https://yt.com/v={video_id}",
        "title": f"Song {video_id}",
        "duration": 100,
        "uploader": "Chan",
    }


class TestResolveConcurrency:
    """PLAY_INFLIGHT_MAX bounds what a guild holds in memory; this bounds what it
    holds of the shared, process-wide yt-dlp pool. Asserted at the EXTRACTION, which
    is where the slot is taken — around the resolve it also caught cache hits."""

    async def test_a_guild_holds_at_most_that_many_workers_at_once(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """Without it one guild's paste burst takes every worker for as many waves
        as it has links, and the jobs queued behind include the playback loop's own
        in-band extractions in OTHER guilds."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.redis = fake_redis
        live = 0
        peak = 0
        release = asyncio.Event()

        async def _extract(_request: Any) -> Any:
            nonlocal live, peak
            live += 1
            peak = max(peak, live)
            await release.wait()
            live -= 1
            return _extracted_song("x")

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLAY_RESOLVE_CONCURRENCY", 2),
            patch("src.youtube._run_extract", new=_extract),
        ):
            tasks = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(
                        music_bot, mock_ctx, url=f"https://yt.com/v=s{n}"
                    )
                )
                for n in range(5)
            ]
            await settle()
            assert peak == 2, peak  # not 5
            release.set()
            await asyncio.gather(*tasks)

        assert mp.queue_put.await_count == 5  # and all of them still land

    async def test_a_cache_hit_does_not_queue_behind_two_extractions(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: aioredis.Redis
    ) -> None:
        """The request class the bound is meant to protect, not to delay: a repeat
        -play needs no worker at all, and around the whole resolve it waited for two
        that were still extracting."""
        mp = mock_mp()
        mock_ctx.voice_client = connected_vc(mock_ctx)
        music_bot.get_mp = MagicMock(return_value=mp)
        music_bot.redis = fake_redis
        await fake_redis.set(
            "ytdl:source:https://yt.com/v=warm",
            orjson.dumps(
                {
                    "webpage_url": "https://yt.com/v=warm",
                    "title": "Already Resolved",
                    "duration": 100,
                    "uploader": "Chan",
                    "thumbnail": None,
                    "cached_at": time.time(),
                }
            ),
        )
        release = asyncio.Event()

        async def _extract(_request: Any) -> Any:
            await release.wait()
            return _extracted_song("x")

        with (
            no_typing("src.commands.play.background_typing"),
            patch("src.play_placement.PLAY_RESOLVE_CONCURRENCY", 2),
            patch("src.youtube._run_extract", new=_extract),
        ):
            holding = [
                asyncio.create_task(
                    command_callback(MusicBot.play)(
                        music_bot, mock_ctx, url=f"https://yt.com/v=cold{n}"
                    )
                )
                for n in range(2)
            ]
            await settle()
            hit = asyncio.create_task(
                command_callback(MusicBot.play)(
                    music_bot, mock_ctx, url="https://yt.com/v=warm"
                )
            )
            await settle()
            assert hit.done()  # both slots held, and it never needed one
            release.set()
            await asyncio.gather(*holding)

        assert mp.queue_put.await_count == 3


class TestRetirePlayerFence:
    """retire_player stamps a player retired without landing mid-placement."""

    async def test_it_waits_for_a_put_in_progress(self, music_bot: MusicBot) -> None:
        """The put writes the deque and then the mirror, and a flag set between the
        two leaves the song in one leg only. So the stamp takes the place lock, and
        a put holding it finishes first."""
        plays = _GuildPlays()
        music_bot._plays._guilds[7] = plays
        mp = MagicMock()
        await plays.lock.acquire()

        retire = asyncio.create_task(music_bot._plays.retire_player(7, cast(Any, mp)))
        for _ in range(3):
            await asyncio.sleep(0)
        mp.mark_retired.assert_not_called()

        plays.lock.release()
        await retire
        mp.mark_retired.assert_called_once()

    async def test_a_stalled_put_does_not_hold_the_stamp_forever(
        self, music_bot: MusicBot
    ) -> None:
        """A stalled Redis must not keep a teardown from retiring the player: past
        the put's own bound the stamp lands anyway, or every later -play places into
        a player that is already torn down."""
        plays = _GuildPlays()
        music_bot._plays._guilds[7] = plays
        mp = MagicMock()
        await plays.lock.acquire()  # never released

        with patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01):
            await music_bot._plays.retire_player(7, cast(Any, mp))

        mp.mark_retired.assert_called_once()
        plays.lock.release()

    async def test_a_guild_with_no_requests_retires_without_a_lock(
        self, music_bot: MusicBot
    ) -> None:
        """Nothing to fence against, so the stamp is immediate."""
        mp = MagicMock()
        await music_bot._plays.retire_player(999, cast(Any, mp))
        mp.mark_retired.assert_called_once()


class TestPlacedMeansLanded:
    """`placed` is what -stop/-clear/-remove read to decide whether a request is
    past dropping. Set before the body, a put that raised claims to have landed."""

    async def test_a_body_that_raises_leaves_the_request_droppable(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        req = admit(music_bot, mock_ctx, mp)

        with pytest.raises(RuntimeError):
            async with music_bot._plays.place(req) as verdict:
                assert verdict.placed
                raise RuntimeError("the put failed")

        assert not req.placed

    async def test_a_body_that_completes_marks_it_placed(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        req = admit(music_bot, mock_ctx, mp)

        async with music_bot._plays.place(req) as verdict:
            assert verdict.placed

        assert req.placed
