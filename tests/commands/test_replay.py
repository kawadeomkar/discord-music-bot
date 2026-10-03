"""Tests for `-replay` (src/commands/replay.py)."""

import time
from collections.abc import Iterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from discord.ext import commands
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from redis.asyncio import Redis

from src import play_pipeline
from src.commands import replay as replay_cmd
from src.commands.replay import (
    NOTHING_TO_REPLAY,
    last_played,
    pending_entry,
    replay_item,
    songs_to_pass_over,
)
from src.guild_history import GuildHistory
from src.guild_queue import matches_origin
from src.guild_state import HistoryEntry
from src.musicbot import MusicBot
from src.musicplayer import InterjectOutcome, MusicPlayer
from src.play_placement import PlaceStalled, play_key
from src.queue_item import QueueObject
from src.youtube import YTDL, StreamProbe
from tests.helpers import (
    command_callback,
    connected_vc,
    loop_song,
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


def _parked(url: str, requester: Any) -> QueueObject:
    return QueueObject(
        webpage_url=url, title="parked", requester=requester, ts=42, is_resume=True
    )


class TestLastPlayed:
    def test_the_newest_entry_is_the_last_song_played(self) -> None:
        history = _history(_entry(_A, "A"), _entry(_B, "B"))
        assert (entry := last_played(history)) is not None
        assert entry.webpage_url == _B

    def test_songs_passed_over_are_skipped_however_many_deep(self) -> None:
        """A -restart of the live song records its fragment, so the newest entries
        can BE the live song; the song the user means is the one before it."""
        history = _history(_entry(_A, "A"), _entry(_B, "B"), _entry(_B, "B again"))
        assert (entry := last_played(history, skip_urls=frozenset({_B}))) is not None
        assert entry.webpage_url == _A

    def test_an_entry_with_no_link_is_passed_over(self) -> None:
        """Zero-values mean unknown on a HistoryEntry, and an empty url has nothing
        for the stream cache or yt-dlp to open. A song is live here, so the empty
        url is not excluded by the skip set by accident."""
        history = _history(_entry(_A, "A"), _entry("", "Lost"))
        entry = last_played(history, skip_urls=frozenset({_C}))
        assert entry is not None
        assert entry.webpage_url == _A

    def test_a_pending_play_is_newer_than_the_cache(self) -> None:
        """The loop writes the row after its prefetch await: until then the song
        that just ended is only `pending`, and it is the newest."""
        history = _history(_entry(_A, "A"))
        assert (entry := last_played(history, pending=_entry(_B, "B"))) is not None
        assert entry.webpage_url == _B

    def test_a_pending_play_is_passed_over_like_any_other(self) -> None:
        history = _history(_entry(_A, "A"))
        entry = last_played(history, pending=_entry(_B, "B"), skip_urls=frozenset({_B}))
        assert entry is not None and entry.webpage_url == _A

    @pytest.mark.parametrize(
        "history",
        [
            pytest.param(_history(), id="empty"),
            pytest.param(_history(_entry(_B, "B")), id="only-the-live-song"),
        ],
    )
    def test_nothing_to_replay(self, history: GuildHistory) -> None:
        assert last_played(history, skip_urls=frozenset({_B})) is None


class TestSongsToPassOver:
    def test_the_live_song_and_every_parked_song(self, mock_ctx: MagicMock) -> None:
        """Without the parked songs a repeated -replay flips between two songs,
        parking each in turn. A queued song that is not parked still counts as a
        candidate: queuing it is not replaying it."""
        mp = mock_mp()
        mp.current_song = MagicMock(webpage_url=_C)
        mp.queue.display_items = MagicMock(
            return_value=[
                _parked(_A, mock_ctx.author),
                QueueObject(webpage_url=_B, title="queued", requester=mock_ctx.author),
            ]
        )
        assert songs_to_pass_over(mp, live=True) == frozenset({_A, _C})

    def test_a_song_that_is_not_live_is_not_passed_over(
        self, mock_ctx: MagicMock
    ) -> None:
        """current_song can name a song already over (the loop has not cleared it):
        only a live one is excluded."""
        mp = mock_mp()
        mp.current_song = MagicMock(webpage_url=_C)
        mp.queue.display_items = MagicMock(return_value=[])
        assert songs_to_pass_over(mp, live=False) == frozenset()


class TestPendingEntry:
    def test_none_when_the_player_has_nothing_unrecorded(self) -> None:
        mp = mock_mp()
        mp.ended_unrecorded = None
        assert pending_entry(mp) is None

    def test_the_row_the_loop_will_write(self) -> None:
        mp = mock_mp()
        mp.guild_id = 77
        mp.ended_unrecorded = loop_song(_B, "Skipped", position=30.0)
        entry = pending_entry(mp)
        assert entry is not None
        assert (entry.webpage_url, entry.title, entry.guild_id) == (_B, "Skipped", 77)


class TestEndedUnrecorded:
    """The window H1 names: current_song is cleared, the history row is not written
    until after the prefetch await, and _ended_song is the only record of the play."""

    def test_an_ended_song_that_will_be_recorded(
        self, music_player: MusicPlayer
    ) -> None:
        song = loop_song(_B, "B", position=30.0)
        music_player._ended_song = song
        assert music_player.ended_unrecorded is song

    def test_a_parked_song_is_not_unrecorded(self, music_player: MusicPlayer) -> None:
        """Its resume tail records it; replaying it would play it twice."""
        song = loop_song(_B, "B", position=30.0)
        music_player._ended_song = song
        music_player._skip_history_for = song
        assert music_player.ended_unrecorded is None

    def test_a_song_nobody_heard_is_not_unrecorded(
        self, music_player: MusicPlayer
    ) -> None:
        """No audio means no row (or a retry): there is nothing to replay."""
        song = loop_song(_B, "B", position=0.0)
        song.produced_audio = False
        music_player._ended_song = song
        assert music_player.ended_unrecorded is None

    def test_nothing_ended(self, music_player: MusicPlayer) -> None:
        assert music_player.ended_unrecorded is None


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
        """A player with A then B in history and C live, so the replay is B."""
        mp = mock_mp()
        mp.guild_id = mock_ctx.guild.id
        mp.history = _history(_entry(_A, "Song A", duration_secs=200), _entry(_B, "B"))
        mp.ended_unrecorded = None
        mp.queue.display_items = MagicMock(return_value=[])
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
        mock_ctx.invoke = AsyncMock()
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
        # Registered as a -play and retired with the reply; a replay spends the
        # cooldown.
        assert not music_bot._plays._guilds
        mock_ctx.command.reset_cooldown.assert_not_called()

    async def test_the_live_songs_own_fragment_is_not_what_replays(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """B is live and its fragment is the newest entry (a -restart wrote it),
        so the song before it replays."""
        mp.current_song.webpage_url = _B

        await self._replay(music_bot, mock_ctx)

        assert (call := mp.interject.await_args) is not None
        assert call.args[0].webpage_url == _A

    async def test_a_skip_still_being_recorded_is_what_replays(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """-skip, then -replay before the loop's prefetch await ends: nothing is
        live, the cache does not hold the skipped song yet, and it is the one the
        user means."""
        mp.current_song = None
        mock_ctx.voice_client = connected_vc(mock_ctx)
        mp.ended_unrecorded = loop_song(_C, "Skipped", position=12.0)

        await self._replay(music_bot, mock_ctx)

        assert (call := mp.queue_put_next.await_args) is not None
        assert call.args[0].webpage_url == _C

    async def test_a_parked_song_is_passed_over(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """B waits to resume (a previous -replay parked it), so it plays again
        anyway: the replay goes one song further back, to A."""
        mp.queue.display_items = MagicMock(return_value=[_parked(_B, mock_ctx.author)])

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

    async def test_a_song_that_ended_during_the_warm_plays_next(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """interject() finds nothing live to park and returns None: the replay
        front-inserts instead, and says so."""
        mp.interject = AsyncMock(return_value=None)

        await self._replay(music_bot, mock_ctx)

        assert (call := mp.queue_put_next.await_args) is not None
        assert call.args[0][0].webpage_url == _B
        embed = self._embed(mock_ctx)
        assert (embed.title or "").startswith("▶️ Playing next: B")
        assert "already ended" in (embed.description or "")

    @pytest.mark.parametrize("join_running", [False, True], ids=["no-voice", "joining"])
    async def test_out_of_voice_it_goes_through_playnext(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        join_running: bool,
    ) -> None:
        """-playnext's cold path owns the join, the gate hold and the teardown when
        either fails, and it never lands the song at the tail or interjects, however
        the join race resolves. Invoked through discord.py, so -playnext's own
        wrapper renders its failures."""
        if join_running:
            mock_ctx.voice_client = connected_vc(mock_ctx)
            music_bot._plays.join_in_flight = MagicMock(return_value=True)
        else:
            mock_ctx.voice_client = None

        await self._replay(music_bot, mock_ctx)

        mock_ctx.invoke.assert_awaited_once_with(music_bot.playnext, url=_B)
        mp.interject.assert_not_awaited()
        mp.queue_put_next.assert_not_awaited()

    async def test_nothing_to_replay_says_so_and_hands_back_the_cooldown(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        mp.history = _history()

        await self._replay(music_bot, mock_ctx)

        embed = self._embed(mock_ctx)
        assert embed.description == NOTHING_TO_REPLAY
        assert embed.color == discord.Color.orange()
        mp.interject.assert_not_awaited()
        assert not music_bot._plays._guilds
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

    async def test_a_restore_still_running_is_reported_and_hands_back_the_cooldown(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mp: MagicMock
    ) -> None:
        """A player this command built is still reading history from Redis: read
        before it lands, the cache is empty and "nothing to replay" would be false."""
        mp.wait_for_restore = AsyncMock(return_value=False)

        await self._replay(music_bot, mock_ctx)

        assert "Still loading" in (self._embed(mock_ctx).description or "")
        mp.interject.assert_not_awaited()
        mock_ctx.command.reset_cooldown.assert_called_once_with(mock_ctx)

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

    async def test_a_remove_of_its_link_during_the_warm_drops_it(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        wired: AsyncMock,
    ) -> None:
        """-remove finds in-flight requests by the query they registered, which is
        the replayed song's link."""

        async def warm_then_remove(
            qo: QueueObject, *, redis: Any = None
        ) -> QueueObject:
            music_bot._plays.inflight(
                play_key(mock_ctx), "remove", lambda r: matches_origin(_B, r.query)
            )
            return qo

        wired.side_effect = warm_then_remove

        await self._replay(music_bot, mock_ctx)

        mp.interject.assert_not_awaited()
        assert "`-remove` ran" in (self._embed(mock_ctx).description or "")

    @pytest.mark.parametrize(
        ("before_the_put", "wording"),
        [(True, "wasn't queued"), (False, "may not have been")],
    )
    async def test_a_stalled_place_says_what_it_may_claim(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        before_the_put: bool,
        wording: str,
    ) -> None:
        with patch.object(
            play_pipeline,
            "interject_resolved",
            new=AsyncMock(side_effect=PlaceStalled(before_the_put=before_the_put)),
        ):
            await self._replay(music_bot, mock_ctx)

        assert wording in (self._embed(mock_ctx).description or "")
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

    @pytest.mark.parametrize(
        ("setup", "key", "value"),
        [
            ("live", "replay.route", "now"),
            ("idle", "replay.route", "next"),
            ("cold", "replay.route", "cold_start"),
            ("empty", "replay.refused", "no_history"),
            ("restoring", "replay.refused", "restore_pending"),
        ],
    )
    async def test_every_exit_names_itself_on_the_span(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        setup: str,
        key: str,
        value: str,
    ) -> None:
        """replay.route and replay.refused are new keys: the old command's
        replay.outcome values moved to restart.outcome, and a query on either must
        not mix the two commands."""
        if setup == "idle":
            mp.current_song = None
            mock_ctx.voice_client = connected_vc(mock_ctx)
        elif setup == "cold":
            mock_ctx.voice_client = None
        elif setup == "empty":
            mp.history = _history()
        elif setup == "restoring":
            mp.wait_for_restore = AsyncMock(return_value=False)
        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        with provider.get_tracer("test").start_as_current_span("bot.replay"):
            await replay_cmd.run(mock_ctx, cog=music_bot)

        (span,) = exporter.get_finished_spans()
        attributes = span.attributes or {}
        assert attributes[key] == value
        assert "replay.outcome" not in attributes
        if setup in ("live", "idle", "cold"):
            assert attributes["replay.url"] == _B


class TestReplayWarmPath:
    """Against a real stream cache: what the gate costs, and what it proves."""

    @pytest.fixture
    def mp(
        self, music_bot: MusicBot, mock_ctx: MagicMock, fake_redis: Redis
    ) -> MagicMock:
        mp = mock_mp()
        mp.guild_id = mock_ctx.guild.id
        mp.history = _history(_entry(_B, "B"))
        mp.ended_unrecorded = None
        mp.queue.display_items = MagicMock(return_value=[])
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
        return mp

    async def _cache(self, redis: Redis, **fields: Any) -> None:
        entry = {
            "webpage_url": _B,
            "title": "B",
            "url": "https://rr1.googlevideo.com/b?expire=9999999999",
            **fields,
        }
        await redis.set(f"ytdl:stream:{_B}", orjson.dumps(entry), ex=1800)

    async def test_a_freshly_probed_song_replays_with_no_probe_and_no_yt_dlp_call(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        fake_redis: Redis,
    ) -> None:
        """The floor: the item is built from the history row, so no search and no
        source-cache read, and a verdict inside _PROBE_REUSE_SECS is not re-checked."""
        await self._cache(fake_redis, probed_at=time.time())

        with (
            patch("src.youtube._ytdlp_extract") as extract,
            patch("src.youtube._probe_stream_url", new=AsyncMock()) as probe,
        ):
            await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        extract.assert_not_called()
        probe.assert_not_awaited()
        mp.interject.assert_awaited_once()

    async def test_a_stale_verdict_is_probed_before_the_interrupt_and_restamped(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        fake_redis: Redis,
    ) -> None:
        """A replayed song's entry is a song old: probed after vc.stop() it costs
        silence, so it is probed while the live song plays, and the fresh stamp lets
        the play reuse the verdict."""
        await self._cache(fake_redis, probed_at=time.time() - 300)
        order: list[str] = []

        async def probe(_url: str) -> StreamProbe:
            order.append("probe")
            return StreamProbe.PLAYABLE

        async def interject(*_a: Any, **_k: Any) -> InterjectOutcome:
            order.append("interject")
            return InterjectOutcome(
                interrupted_title="Song C", resume_position=95, was_paused=False
            )

        mp.interject = AsyncMock(side_effect=interject)
        with (
            patch("src.youtube._ytdlp_extract") as extract,
            patch("src.youtube._probe_stream_url", new=AsyncMock(side_effect=probe)),
        ):
            await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        assert order == ["probe", "interject"]
        extract.assert_not_called()
        raw = await fake_redis.get(f"ytdl:stream:{_B}")
        assert raw is not None
        assert time.time() - orjson.loads(raw)["probed_at"] < 5

    async def test_a_revoked_url_that_cannot_be_replaced_stops_nothing(
        self,
        music_bot: MusicBot,
        mock_ctx: MagicMock,
        mp: MagicMock,
        fake_redis: Redis,
    ) -> None:
        """A dead cached URL is re-extracted before the interrupt; one that cannot
        be fails the command and leaves the live song playing."""
        await self._cache(fake_redis, probed_at=time.time() - 300)
        music_bot._command_error = AsyncMock()

        with (
            patch("src.youtube._ytdlp_extract", return_value=None),
            patch(
                "src.youtube._probe_stream_url",
                new=AsyncMock(return_value=StreamProbe.DEAD),
            ),
        ):
            await command_callback(MusicBot.replay)(music_bot, mock_ctx)

        mp.interject.assert_not_awaited()
        music_bot._command_error.assert_awaited_once()


class TestConfirmStream:
    async def test_without_redis_there_is_nothing_to_prove(
        self, mock_author: MagicMock
    ) -> None:
        item = QueueObject(webpage_url=_B, title="B", requester=mock_author)
        assert await YTDL.confirm_stream(item, redis=None) is True

    async def test_without_an_entry_there_is_nothing_to_prove(
        self, mock_author: MagicMock, fake_redis: Redis
    ) -> None:
        """prefetch_stream already extracted and probed a miss."""
        item = QueueObject(webpage_url=_B, title="B", requester=mock_author)
        with patch("src.youtube._probe_stream_url", new=AsyncMock()) as probe:
            assert await YTDL.confirm_stream(item, redis=fake_redis) is True
        probe.assert_not_awaited()


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

    def test_replay_is_rate_limited_per_guild(self) -> None:
        """Each repeat parks another song; the cooldown bounds how fast they stack."""
        buckets = MusicBot.replay._buckets
        assert buckets.valid, "-replay lost its cooldown"
        assert buckets._cooldown is not None
        assert (buckets._cooldown.rate, buckets._cooldown.per) == (1, 5.0)
        assert buckets.type is commands.BucketType.guild

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
