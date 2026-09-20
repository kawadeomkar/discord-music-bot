"""Tests for src/play_placement.py — the `-play` placement grammar, and the
registry whose place lock makes "check, then insert" atomic."""

import asyncio
import contextlib
import time
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import orjson
import redis.asyncio as aioredis

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
import discord
from discord.ext import commands

from src.musicbot import MusicBot
from src import util
from src.util import channel_claim
from src import config
from src.play_placement import (
    PlaceStalled,
    PlayArgs,
    _GuildPlays,
    _NOTICE_CLAIM,
    PlayMode,
    ResolveWaitExpired,
    play_key,
    slow_resolve_notice,
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
        config.set_override("PLAY_INFLIGHT_MAX", 2)
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
        config.set_override("PLAY_INFLIGHT_MAX", 1)
        admit(music_bot, mock_ctx, mp)
        admit(music_bot, other, mp)  # no raise

    def test_the_drop_stamp_signals_each_request_it_names(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        wanted = admit(music_bot, mock_ctx, mp, query="wanted")
        kept = admit(music_bot, mock_ctx, mp, query="kept")
        placed = admit(music_bot, mock_ctx, mp, query="wanted")
        placed.placed = True

        music_bot._plays.inflight(
            play_key(mock_ctx), "remove", lambda r: r.query == "wanted"
        )

        assert wanted.settled.is_set()
        assert not kept.settled.is_set()
        assert not placed.settled.is_set()

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
        config.set_override("PLAY_INFLIGHT_MAX", 1)
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
    """The in-flight cap bounds what a guild holds in memory; this bounds what it
    holds of the shared, process-wide yt-dlp pool. Asserted at the EXTRACTION, which
    is where the slot is taken — around the resolve it also caught cache hits."""

    def test_a_change_applies_once_the_guild_is_rebuilt(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A built semaphore cannot be resized, so a guild with requests in flight
        keeps the bound it was built with, and takes the new one once idle."""
        mp = mock_mp()
        config.set_override("PLAY_RESOLVE_CONCURRENCY", 1)
        first = admit(music_bot, mock_ctx, mp)
        config.set_override("PLAY_RESOLVE_CONCURRENCY", 3)
        second = admit(music_bot, mock_ctx, mp)
        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        assert plays.resolves._value == 1
        music_bot._plays.retire(first)
        music_bot._plays.retire(second)
        admit(music_bot, mock_ctx, mp)
        assert music_bot._plays._guilds[play_key(mock_ctx)].resolves._value == 3

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

        config.set_override("PLAY_RESOLVE_CONCURRENCY", 2)
        with (
            no_typing("src.commands.play.background_typing"),
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

        config.set_override("PLAY_RESOLVE_CONCURRENCY", 2)
        with (
            no_typing("src.commands.play.background_typing"),
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

    async def test_it_signals_every_request_the_retired_player_will_refuse(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """A kick or the alone-watchdog tears the player down without calling
        inflight(), so the retire is where their cards and notices learn it."""
        mp, other = mock_mp(), mock_mp()
        resolving = admit(music_bot, mock_ctx, mp)
        placed = admit(music_bot, mock_ctx, mp)
        placed.placed = True
        elsewhere = admit(music_bot, mock_ctx, other)

        await music_bot._plays.retire_player(play_key(mock_ctx), cast(Any, mp))

        assert resolving.settled.is_set()
        assert not placed.settled.is_set()
        assert not elsewhere.settled.is_set()


class TestPlacedMeansLanded:
    """`placed` is what -stop/-clear/-remove read to decide whether a request is
    past dropping. It is set once the checks pass, so a command arriving during the
    put leaves the request alone, and a put that raised gives the request back."""

    async def test_a_request_is_placed_while_its_put_runs(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        music_bot.get_mp = MagicMock(return_value=mp)
        req = admit(music_bot, mock_ctx, mp)

        async with music_bot._plays.place(req) as verdict:
            assert verdict.placed
            assert req.placed
            stopped = music_bot._plays.inflight(play_key(mock_ctx), "remove")
            # place() sets it on return too, so a drop is only visible in here.
            assert not req.settled.is_set()

        assert stopped == []
        assert req.dropped_by == ""

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


class TestPlaceSettlesTheRequest:
    """`settled` is what takes a request's card or notice down. place() sets it on
    the way out, so the message is gone before the reply — whose sends and
    reactions can outlast the delay it waited out — rather than after it."""

    async def test_a_placed_request_settles_once_the_put_returns(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        mp = mock_mp()
        req = admit(music_bot, mock_ctx, mp)

        async with music_bot._plays.place(req) as verdict:
            assert verdict.placed
            # The put is still running: the card is still true.
            assert not req.settled.is_set()

        assert req.settled.is_set()

    async def test_a_refused_request_settles_too(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Its reply is the "dropped" report, which is no faster than a
        confirmation."""
        mp = mock_mp()
        req = admit(music_bot, mock_ctx, mp)
        req.dropped_by = "stop"

        async with music_bot._plays.place(req) as verdict:
            assert not verdict.placed

        assert req.settled.is_set()
        assert not req.placed

    async def test_a_stalled_place_settles_the_request(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        req = admit(music_bot, mock_ctx, mock_mp())
        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        await plays.lock.acquire()
        try:
            with (
                patch("src.play_placement.PLACE_TIMEOUT_SECS", 0.01),
                pytest.raises(PlaceStalled),
            ):
                async with music_bot._plays.place(req):
                    pass  # pragma: no cover - the lock is never acquired
        finally:
            plays.lock.release()

        assert req.settled.is_set()


class TestResolveSlot:
    """The guild's resolve bound, with a deadline on the WAIT for it. The
    extraction it guards stays unbounded — the slot exists to queue expensive work,
    so a bound covering it would cancel the resolve it was sized for."""

    async def test_a_free_slot_is_entered_without_waiting(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        req = admit(music_bot, mock_ctx, mock_mp())
        slot = music_bot._plays.resolve_slot(req)
        async with slot:
            pass  # released on exit; a second entry proves it

        async with music_bot._plays.resolve_slot(req):
            pass

    async def test_the_extraction_holding_a_slot_is_not_bounded(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """The whole point of the split: a 5,547-track playlist runs 99s inside the
        slot and must not be cut off by the bound on queueing FOR one."""
        req = admit(music_bot, mock_ctx, mock_mp())
        config.set_override("PLAY_RESOLVE_WAIT_SECS", 0.05)
        async with music_bot._plays.resolve_slot(req):
            # Comfortably past the wait bound, inside the slot.
            await asyncio.sleep(0.15)

    async def test_a_slot_that_never_frees_expires_the_wait(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        req = admit(music_bot, mock_ctx, mock_mp())
        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        # Take every slot and never give one back.
        for _ in range(config.play_resolve_concurrency()):
            await plays.resolves.acquire()

        config.set_override("PLAY_RESOLVE_WAIT_SECS", 0.05)
        with (
            recording_span() as span,
            pytest.raises(ResolveWaitExpired),
        ):
            async with music_bot._plays.resolve_slot(req):
                pass  # pragma: no cover - the acquire above never returns
        span.set_attribute.assert_any_call("play.resolve_wait_expired", True)

    async def test_an_expired_wait_releases_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """__aexit__ does not run for an __aenter__ that raised. A release here
        would hand out a permit the guild never held, uncapping the bound."""
        req = admit(music_bot, mock_ctx, mock_mp())
        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        for _ in range(config.play_resolve_concurrency()):
            await plays.resolves.acquire()

        config.set_override("PLAY_RESOLVE_WAIT_SECS", 0.05)
        with pytest.raises(ResolveWaitExpired):
            async with music_bot._plays.resolve_slot(req):
                pass  # pragma: no cover
        assert plays.resolves.locked()

    async def test_the_expiry_names_the_wait_that_ran(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """Read once on entry: a change landing mid-wait neither stretches this
        wait nor makes its message quote a bound it never had."""
        req = admit(music_bot, mock_ctx, mock_mp())
        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        for _ in range(config.play_resolve_concurrency()):
            await plays.resolves.acquire()
        config.set_override("PLAY_RESOLVE_WAIT_SECS", 0.05)

        async def _wait() -> None:
            async with music_bot._plays.resolve_slot(req):
                pass  # pragma: no cover - never acquired

        waiting = asyncio.create_task(_wait())
        await asyncio.sleep(0)
        config.set_override("PLAY_RESOLVE_WAIT_SECS", 300.0)
        with pytest.raises(ResolveWaitExpired, match=r"within 0\.05s"):
            async with asyncio.timeout(2):
                await waiting

    async def test_a_slot_freed_inside_the_bound_is_taken(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        req = admit(music_bot, mock_ctx, mock_mp())
        plays = music_bot._plays._guilds[play_key(mock_ctx)]
        for _ in range(config.play_resolve_concurrency()):
            await plays.resolves.acquire()

        async def _free() -> None:
            await asyncio.sleep(0.02)
            plays.resolves.release()

        freeing = asyncio.create_task(_free())
        config.set_override("PLAY_RESOLVE_WAIT_SECS", 5.0)
        async with music_bot._plays.resolve_slot(req):
            entered = True
        await freeing
        assert entered


class TestResolveSlotTelemetry:
    async def test_a_successful_acquire_records_what_it_waited(
        self, music_bot: MusicBot, mock_ctx: MagicMock
    ) -> None:
        """play.resolve_wait_secs is the number PLAY_RESOLVE_CONCURRENCY is
        tuned from — it is emitted on EVERY acquire, not only a slow one, which
        is what makes a burst diagnosable after the fact rather than only while
        it is happening."""
        with patch("src.play_placement.trace.get_current_span") as current:
            span = current.return_value
            req = admit(music_bot, mock_ctx, mock_mp())
            async with music_bot._plays.resolve_slot(req):
                pass
        names = [call.args[0] for call in span.set_attribute.call_args_list]
        assert "play.resolve_wait_secs" in names


class TestSlowResolveNotice:
    """A request that outlives the delay says so, and takes the message back when
    it lands. Silence is what the concurrent resolve costs the user: nothing is
    serialized, so there is no queue position to report — only that work is on."""

    async def test_a_fast_resolve_says_nothing(self, mock_ctx: MagicMock) -> None:
        async with slow_resolve_notice(mock_ctx, query="a song", delay=5.0):
            pass
        mock_ctx.channel.send.assert_not_awaited()

    async def test_a_slow_resolve_posts_and_retracts(self, mock_ctx: MagicMock) -> None:
        async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
            await asyncio.sleep(0.08)
        mock_ctx.channel.send.assert_awaited_once()
        mock_ctx.channel.send.return_value.delete.assert_awaited_once()

    async def test_the_notice_never_becomes_the_now_playing_host(
        self, mock_ctx: MagicMock
    ) -> None:
        """ctx.channel.send, not ctx.send: MusicContext.send would adopt this as the
        NP host, and deleting it would drag the live progress bar onto it."""
        async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
            await asyncio.sleep(0.08)
        mock_ctx.send.assert_not_awaited()

    async def test_a_send_that_fails_leaves_the_request_alone(
        self, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.channel.send.side_effect = discord.HTTPException(
            MagicMock(), "no perms"
        )
        async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
            await asyncio.sleep(0.08)

    async def test_off_arms_nothing(self, mock_ctx: MagicMock) -> None:
        """A server that turned the notice off gets no poster task, not one whose
        post never comes."""
        before = asyncio.all_tasks()
        async with slow_resolve_notice(mock_ctx, query="a song", delay=None):
            assert asyncio.all_tasks() == before
            await asyncio.sleep(0.05)
        mock_ctx.channel.send.assert_not_awaited()

    async def test_the_notice_carries_the_debug_footer(
        self, mock_ctx: MagicMock
    ) -> None:
        """It sends through ctx.channel.send, which bypasses the decoration
        MusicContext.send applies — so like the two dashboards and the card, the
        footer has to be threaded in."""
        async with slow_resolve_notice(
            mock_ctx, query="a song", delay=0.02, debug_suffix="trace=abc"
        ):
            await asyncio.sleep(0.08)
        embed = mock_ctx.channel.send.await_args.kwargs["embed"]
        assert embed.footer.text == "trace=abc"

    async def test_no_footer_means_no_footer(self, mock_ctx: MagicMock) -> None:
        async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
            await asyncio.sleep(0.08)
        embed = mock_ctx.channel.send.await_args.kwargs["embed"]
        assert embed.footer.text is None

    async def test_a_burst_in_one_channel_posts_one_notice(
        self, mock_ctx: MagicMock
    ) -> None:
        """PLAY_INFLIGHT_MAX is 16 and PLAY_RESOLVE_CONCURRENCY is 2, so the tail
        of a pasted burst is GUARANTEED to cross the delay. Sixteen notices at once
        would sit on the channel's create bucket ahead of the confirmations the
        user asked for."""

        async def _one() -> None:
            async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
                await asyncio.sleep(0.08)

        await asyncio.gather(*(_one() for _ in range(16)))
        mock_ctx.channel.send.assert_awaited_once()
        mock_ctx.channel.send.return_value.delete.assert_awaited_once()

    async def test_the_notice_names_its_request(self, mock_ctx: MagicMock) -> None:
        """Unattributed, B reads A's notice as theirs, and it disappears when A
        lands while B is still resolving."""
        mock_ctx.author.mention = "<@42>"
        async with slow_resolve_notice(mock_ctx, query="daft *punk*", delay=0.02):
            await asyncio.sleep(0.08)
        description = mock_ctx.channel.send.await_args.kwargs["embed"].description
        assert "<@42>" in description
        assert "daft \\*punk\\*" in description  # escaped, not styled

    async def test_a_notice_that_lost_the_channel_posts_once_it_is_free(
        self, mock_ctx: MagicMock
    ) -> None:
        shown: list[int] = []

        async def _one(secs: float) -> None:
            async with slow_resolve_notice(mock_ctx, query=f"{secs}", delay=0.02):
                await asyncio.sleep(secs)
                shown.append(mock_ctx.channel.send.await_count)

        with (
            patch("src.play_placement._NOTICE_RETRY_SECS", 0.01),
        ):
            await asyncio.gather(_one(0.06), _one(0.3))

        assert shown == [1, 2]

    @pytest.mark.parametrize("outcome", ["shown", "send_failed", "claimed_elsewhere"])
    async def test_the_span_says_what_the_notice_did(
        self, mock_ctx: MagicMock, outcome: str
    ) -> None:
        if outcome == "send_failed":
            mock_ctx.channel.send.side_effect = discord.HTTPException(
                MagicMock(status=403), "Missing Permissions"
            )
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        with contextlib.ExitStack() as held:
            if outcome == "claimed_elsewhere":
                held.enter_context(channel_claim(_NOTICE_CLAIM, mock_ctx.channel.id))
            with (
                provider.get_tracer("test").start_as_current_span("bot.play"),
            ):
                async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
                    await asyncio.sleep(0.08)

        (span,) = exporter.get_finished_spans()
        assert (span.attributes or {})["play.slow_notice"] == outcome

    async def test_a_notice_and_a_card_are_different_kinds(
        self, mock_ctx: MagicMock
    ) -> None:
        """One claim per KIND: a channel already showing a card for someone's
        playlist must still be able to say a different request is slow."""
        with channel_claim("queue-progress-card", mock_ctx.channel.id) as card:
            assert card
            async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
                await asyncio.sleep(0.08)
        mock_ctx.channel.send.assert_awaited_once()

    async def test_a_fast_resolve_does_not_hold_the_slot(
        self, mock_ctx: MagicMock
    ) -> None:
        """Claimed at the send, not at entry: a request that settles inside the
        delay never sends, and must not deny the slot to a slow sibling."""

        async def _fast() -> None:
            async with slow_resolve_notice(mock_ctx, query="a song", delay=0.05):
                await asyncio.sleep(0.01)

        async def _slow() -> None:
            async with slow_resolve_notice(mock_ctx, query="a song", delay=0.05):
                await asyncio.sleep(0.2)

        await asyncio.gather(_fast(), _slow())
        mock_ctx.channel.send.assert_awaited_once()

    async def test_a_cancel_while_joining_the_notice_propagates_and_retracts(
        self, mock_ctx: MagicMock
    ) -> None:
        """Cancelling a -play whose notice is mid-send must not be swallowed —
        a command that returns normally from its own cancellation stalls the
        shutdown gather that cancelled it. The task that sent the notice takes
        it back in its own finally, so the message does not outlive the cancel
        either."""
        message = mock_ctx.channel.send.return_value
        sending, release = asyncio.Event(), asyncio.Event()

        async def _send(**_: Any) -> MagicMock:
            sending.set()
            await release.wait()
            return message

        mock_ctx.channel.send = AsyncMock(side_effect=_send)

        async def _body() -> None:
            async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
                async with asyncio.timeout(2):
                    await sending.wait()

        task = asyncio.create_task(_body())
        async with asyncio.timeout(2):
            await sending.wait()
        # The body has returned; the context manager is parked on the join.
        await settle()
        task.cancel()
        await settle()
        release.set()
        await settle()

        with pytest.raises(asyncio.CancelledError):
            await task
        message.delete.assert_awaited_once()

    async def test_the_body_raising_still_retracts_the_notice(
        self, mock_ctx: MagicMock
    ) -> None:
        with pytest.raises(RuntimeError):
            async with slow_resolve_notice(mock_ctx, query="a song", delay=0.02):
                await asyncio.sleep(0.08)
                raise RuntimeError("resolve failed")
        mock_ctx.channel.send.return_value.delete.assert_awaited_once()

    async def test_a_dropped_request_retracts_before_its_resolve_returns(
        self, mock_ctx: MagicMock
    ) -> None:
        """-stop/-clear/-remove stamp the request mid-resolve, and place() reads
        the stamp only once the resolve returns: the notice would otherwise still
        say the song "will be queued" beside the reply saying it was dropped."""
        dropped = asyncio.Event()
        message = mock_ctx.channel.send.return_value
        async with slow_resolve_notice(
            mock_ctx, query="a song", delay=0.02, request_settled=dropped
        ):
            await asyncio.sleep(0.05)
            mock_ctx.channel.send.assert_awaited_once()
            dropped.set()
            await asyncio.sleep(0.02)
            message.delete.assert_awaited_once()
            assert not util._CLAIMED_CHANNELS.get(_NOTICE_CLAIM)
        message.delete.assert_awaited_once()

    async def test_a_request_dropped_inside_the_delay_posts_nothing(
        self, mock_ctx: MagicMock
    ) -> None:
        dropped = asyncio.Event()
        async with slow_resolve_notice(
            mock_ctx, query="a song", delay=0.05, request_settled=dropped
        ):
            dropped.set()
            await asyncio.sleep(0.1)
        mock_ctx.channel.send.assert_not_awaited()
