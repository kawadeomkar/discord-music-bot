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
    NEXT_FLAG,
    NOW_FLAG,
    TIMESTAMP_FLAG,
    _OPTIONS,
    _PLAY_OPTIONS,
    play_usage,
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
from src.sources import MAX_START_OFFSET_SECS, START_OFFSET_FORMATS
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
    """The leading options come off -play's argument, or none do: the remainder is
    both the search and the origin `-remove` matches on, so a flag stripped from
    mid-line would leave a value the user never typed."""

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
            # A flag past the leading run stays in the text.
            ("song --now", PlayMode.NORMAL, "song --now"),
            ("song --next", PlayMode.NORMAL, "song --next"),
            ("", PlayMode.NORMAL, ""),
            ("   ", PlayMode.NORMAL, ""),
        ],
    )
    def test_the_head_decides(self, argument: str, mode: PlayMode, query: str) -> None:
        args = split_play_args(argument)
        assert (args.mode, args.query) == (mode, query)
        assert (args.dash_typo, args.error, args.start_offset) == (None, None, None)

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


class TestTheTimestampFlag:
    """`--timestamp <time>` joins the leading run: at most one, alongside at most
    one placement flag, in either order."""

    @pytest.mark.parametrize("flag", [TIMESTAMP_FLAG, "--ts", "-ts"])
    def test_every_spelling_takes_the_time(self, flag: str) -> None:
        """`-ts` is the short form; `--ts` is what someone aiming between the two
        writes, and all three reach the same parser."""
        args = split_play_args(f"{flag} 1:32 never gonna give you up")
        assert (args.mode, args.start_offset, args.query) == (
            PlayMode.NORMAL,
            92,
            "never gonna give you up",
        )

    @pytest.mark.parametrize("sep", [" ", "="])
    def test_the_value_may_be_joined_or_separate(self, sep: str) -> None:
        args = split_play_args(f"--ts{sep}1:32 song")
        assert (args.start_offset, args.query) == (92, "song")

    def test_an_equals_with_the_time_in_the_next_token(self) -> None:
        """`--ts= 1:32` is a space away from the joined form, and a time does
        follow it."""
        args = split_play_args("--ts= 1:32 song")
        assert (args.start_offset, args.query, args.error) == (92, "song", None)

    @pytest.mark.parametrize(
        "argument,mode,offset,query",
        [
            ("--now --ts 1:32 song", PlayMode.NOW, 92, "song"),
            ("--ts 1:32 --now song", PlayMode.NOW, 92, "song"),
            ("--next --ts 43 song", PlayMode.NEXT, 43, "song"),
            (
                "-ts 2:04:30 --next https://youtu.be/x",
                PlayMode.NEXT,
                7470,
                "https://youtu.be/x",
            ),
            ("--TS 1:32 --NOW song", PlayMode.NOW, 92, "song"),
        ],
    )
    def test_it_combines_with_a_placement_in_either_order(
        self, argument: str, mode: PlayMode, offset: int, query: str
    ) -> None:
        args = split_play_args(argument)
        assert (args.mode, args.start_offset, args.query) == (mode, offset, query)
        assert args.error is None

    @pytest.mark.parametrize(
        "argument",
        [
            "--ts 1:32 --ts 2:00 song",
            "--now --now song",
            "--next --next song",
            "--now --ts 1:32 --now song",
        ],
    )
    def test_a_repeated_option_is_answered(self, argument: str) -> None:
        """Searching YouTube for the user's own flag is the one outcome that is
        certainly wrong, and a second value silently losing to the first is the
        other."""
        args = split_play_args(argument)
        assert args.error is not None
        assert "twice" in args.error
        assert args.start_offset is None

    @pytest.mark.parametrize("argument", ["--now --next song", "--next --now song"])
    def test_the_two_placements_do_not_combine(self, argument: str) -> None:
        args = split_play_args(argument)
        assert args.error is not None
        assert NOW_FLAG in args.error and NEXT_FLAG in args.error

    @pytest.mark.parametrize("argument", ["--ts", "--ts song", "--now --ts"])
    def test_a_time_that_is_missing_or_unreadable_is_answered(
        self, argument: str
    ) -> None:
        args = split_play_args(argument)
        assert args.error is not None
        assert args.start_offset is None

    def test_an_unreadable_time_names_the_shapes_it_takes(self) -> None:
        args = split_play_args("--ts banana song")
        assert args.error is not None
        assert "banana" in args.error
        assert START_OFFSET_FORMATS in args.error

    def test_a_time_longer_than_a_day_is_answered(self) -> None:
        """Unbounded, a digit run backdates the play epoch by geological spans and
        renders an echo past Discord's description limit."""
        args = split_play_args(f"--ts {MAX_START_OFFSET_SECS} song")
        assert args.error is not None
        assert split_play_args(f"--ts {MAX_START_OFFSET_SECS - 1} song").error is None

    def test_the_echoed_value_cannot_break_out_of_its_code_span_or_run_long(
        self,
    ) -> None:
        """The value is the user's own text on its way into a code span in an
        embed, so it is capped and neutralized like every other echo."""
        args = split_play_args("--ts `x`[y](z) song")
        assert args.error is not None
        assert "`x`" not in args.error and "[y](z)" not in args.error
        long = split_play_args(f"--ts {'9' * 4000} song")
        assert long.error is not None and len(long.error) < 300

    @pytest.mark.parametrize(
        "argument,mode,query",
        [
            ("song --ts 1:32", PlayMode.NORMAL, "song --ts 1:32"),
            (
                "never gonna --ts give you up",
                PlayMode.NORMAL,
                "never gonna --ts give you up",
            ),
            ("--now song --ts 1:32", PlayMode.NOW, "song --ts 1:32"),
            ("--ts 43 song --now", PlayMode.NORMAL, "song --now"),
        ],
    )
    def test_a_flag_past_the_leading_run_is_search_text(
        self, argument: str, mode: PlayMode, query: str
    ) -> None:
        """The same rule the placement flags follow: only the run at the front is
        parsed, so a search keeping its own words stays removable by them."""
        args = split_play_args(argument)
        assert (args.mode, args.query, args.error) == (mode, query, None)

    def test_a_dash_away_from_the_long_form_asks(self) -> None:
        """`-ts` is a real spelling, so only a dash Discord replaced can reach the
        did-you-mean — and it names the long form."""
        assert split_play_args("–ts 1:32 song").dash_typo == TIMESTAMP_FLAG
        assert split_play_args("-timestamp 1:32 song").dash_typo == TIMESTAMP_FLAG
        assert split_play_args("-ts 1:32 song").dash_typo is None

    def test_a_near_miss_inside_the_run_still_asks(self) -> None:
        args = split_play_args("--ts 1:32 -now song")
        assert args.dash_typo == NOW_FLAG
        assert args.query == "--ts 1:32 -now song"

    def test_a_refusal_queues_nothing_and_carries_no_offset(self) -> None:
        """`error` and `start_offset` are exclusive: a half-parsed run must not
        leave an offset a caller could still apply."""
        args = split_play_args("--ts 1:32 --ts 2:00 song")
        assert (args.start_offset, args.mode) == (None, PlayMode.NORMAL)

    def test_the_query_keeps_its_internal_whitespace(self) -> None:
        """The query is the origin -remove matches on, so the flag path and the
        no-flag path have to leave the same string behind."""
        assert split_play_args("-ts 43  a   b").query == "a   b"
        assert split_play_args("a   b").query == "a   b"

    def test_a_quoted_search_survives_the_split(self) -> None:
        """read_rest hands the quotes through and the command unquotes what the
        split returns — an origin that keeps them is one -remove cannot match."""
        assert split_play_args('-ts 43 "never gonna give you up"').query == (
            '"never gonna give you up"'
        )

    def test_a_bare_zero_is_an_offset_not_an_absence(self) -> None:
        """0 is falsy and `start_offset is not None` is what every consumer reads;
        a truthiness test here would silently drop the flag's own refusals."""
        args = split_play_args("--ts 0 song")
        assert args.start_offset == 0

    def test_nothing_to_play_leaves_an_empty_query(self) -> None:
        """The command reports a missing argument; the split just says there is
        none, so both spellings of "no song" answer the same way."""
        args = split_play_args("--ts 1:32")
        assert (args.start_offset, args.query, args.error) == (92, "", None)


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
        config.play_inflight_max.set_override(2)
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
        config.play_inflight_max.set_override(1)
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
        config.play_inflight_max.set_override(1)
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
        config.play_resolve_concurrency.set_override(1)
        first = admit(music_bot, mock_ctx, mp)
        config.play_resolve_concurrency.set_override(3)
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

        config.play_resolve_concurrency.set_override(2)
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

        config.play_resolve_concurrency.set_override(2)
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
        config.play_resolve_wait_secs.set_override(0.05)
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

        config.play_resolve_wait_secs.set_override(0.05)
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

        config.play_resolve_wait_secs.set_override(0.05)
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
        config.play_resolve_wait_secs.set_override(0.05)

        async def _wait() -> None:
            async with music_bot._plays.resolve_slot(req):
                pass  # pragma: no cover - never acquired

        waiting = asyncio.create_task(_wait())
        await asyncio.sleep(0)
        config.play_resolve_wait_secs.set_override(300.0)
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
        config.play_resolve_wait_secs.set_override(5.0)
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


class TestTheOptionRegistry:
    """Every branch of split_play_args reads `_PLAY_OPTIONS`, so an option added
    there is parsed, refused, suggested and documented without editing the parser.
    These pin that, rather than the three options that happen to be registered."""

    def test_every_spelling_reaches_its_option(self) -> None:
        for option in _PLAY_OPTIONS:
            for spelling in option.spellings:
                assert _OPTIONS[spelling] is option

    def test_every_option_is_recognised_by_every_spelling(self) -> None:
        """The table is the only place a spelling is declared, so a spelling it
        carries must parse rather than fall through to the search."""
        for option in _PLAY_OPTIONS:
            for spelling in option.spellings:
                text = f"{spelling} 1:32 song" if option.read else f"{spelling} song"
                args = split_play_args(text)
                assert args.error is None, (spelling, args.error)
                assert args.query == "song"

    def test_an_option_either_stands_for_a_value_or_reads_one(self) -> None:
        """`read` is what decides whether the next token is consumed, so an entry
        carrying both would silently eat a word of the search."""
        for option in _PLAY_OPTIONS:
            assert (option.constant is None) != (option.read is None), option.name

    def test_only_a_value_taking_option_has_a_placeholder(self) -> None:
        for option in _PLAY_OPTIONS:
            assert bool(option.placeholder) is (option.read is not None), option.name

    @pytest.mark.parametrize("option", _PLAY_OPTIONS, ids=lambda o: o.name)
    def test_every_option_refuses_its_own_repeat(self, option: Any) -> None:
        text = (
            f"{option.name} 1:32 {option.name} 1:32 s"
            if option.read
            else (f"{option.name} {option.name} s")
        )
        args = split_play_args(text)
        assert args.error is not None and "twice" in args.error
        assert option.name in args.error

    @pytest.mark.parametrize("option", _PLAY_OPTIONS, ids=lambda o: o.name)
    def test_a_near_miss_of_every_option_is_suggested(self, option: Any) -> None:
        """One dash short, or the em dash iOS substitutes — each names the long
        form, built from the registry rather than a second list."""
        for spelling in option.spellings:
            stem = spelling.lstrip("-")
            assert split_play_args(f"—{stem} x").dash_typo == option.name

    @pytest.mark.parametrize("option", _PLAY_OPTIONS, ids=lambda o: o.name)
    def test_a_valueless_option_refuses_a_value_and_the_reverse(
        self, option: Any
    ) -> None:
        """`--now=x` was searched for as text before the registry; both shapes are
        now answered by one rule that names the option it is about."""
        args = split_play_args(f"{option.name}=x song")
        assert args.error is not None and option.name in args.error
        if option.read is None:
            assert "doesn't take a value" in args.error
        else:
            # It takes one, so `=x` is read as the value and refused on its own
            # terms rather than for carrying one at all.
            assert "doesn't take a value" not in args.error

    def test_options_over_one_field_are_mutually_exclusive(self) -> None:
        """Two options naming the same field are alternatives; the message lists
        that field's options in registry order, so it reads the same whichever was
        typed first."""
        first = split_play_args(f"{NOW_FLAG} {NEXT_FLAG} s").error
        second = split_play_args(f"{NEXT_FLAG} {NOW_FLAG} s").error
        assert first is not None and first == second
        assert NOW_FLAG in first and NEXT_FLAG in first

    def test_options_over_different_fields_combine(self) -> None:
        """Only a shared field makes two options exclusive, so every cross-field
        pair has to survive in both orders."""
        for a in _PLAY_OPTIONS:
            for b in _PLAY_OPTIONS:
                if a.field == b.field:
                    continue
                spelled = [f"{o.name} 1:32" if o.read else o.name for o in (a, b)]
                args = split_play_args(f"{' '.join(spelled)} song")
                assert args.error is None, (a.name, b.name, args.error)
                assert args.query == "song"

    def test_the_usage_line_names_every_option(self) -> None:
        """`-play`'s missing-argument notice renders from the registry, so a new
        option documents itself there."""
        usage = play_usage()
        for option in _PLAY_OPTIONS:
            assert option.name in usage
            assert (option.placeholder in usage) if option.placeholder else True
        assert usage.startswith("play ") and usage.endswith("<url|search>")

    def test_alternatives_share_one_bracket(self) -> None:
        """`[--now|--next]` rather than two brackets: they are one choice."""
        assert f"[{NOW_FLAG}|{NEXT_FLAG}]" in play_usage()
        assert f"[{TIMESTAMP_FLAG} <time>]" in play_usage()

    def test_the_usage_hint_on_a_refusal_is_the_options_own(self) -> None:
        """A `--now` mistake used to be answered with a `--timestamp` example."""
        placement = split_play_args(f"{NOW_FLAG}=x song").error
        assert placement is not None
        assert NOW_FLAG in placement and TIMESTAMP_FLAG not in placement
        timed = split_play_args(TIMESTAMP_FLAG).error
        assert timed is not None and "1:32" in timed
