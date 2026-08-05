"""Tests for src/guild_state.py — field constants and value objects."""

import dataclasses
import logging
from types import SimpleNamespace
from typing import Any, cast

import discord
import orjson
import pytest

from src.sources import YTSource
from src.youtube import YTDL, QueueObject
from src.guild_state import (
    ConfigField,
    GuildConfig,
    GuildPlaybackSnapshot,
    GuildRecoveryGate,
    GuildStateData,
    HistoryEntry,
    NowPlayingData,
    SearchQueueEntry,
    SongQueueEntry,
    parse_history_entry,
    parse_queue_entry,
    serialize_history_entry,
)


def _full_state_hash() -> dict[bytes, bytes]:
    return {
        b"volume": b"0.5",
        b"voice_channel_id": b"111",
        b"text_channel_id": b"222",
        b"current_song_url": b"https://youtu.be/abc",
        b"current_song_title": b"Test Song",
        b"current_song_duration": b"240",
        b"current_song_uploader": b"Test Channel",
        b"current_song_requester_id": b"333",
        b"play_start_epoch": b"1000.5",
        b"total_pause_seconds": b"12.5",
        b"pause_start_epoch": b"1100.0",
    }


class TestGuildStateDataFromRedis:
    def test_full_hash_parses_all_fields(self) -> None:
        data = GuildStateData.from_redis(_full_state_hash())
        assert data.volume == 0.5
        assert data.voice_channel_id == 111
        assert data.text_channel_id == 222
        assert data.current_song_url == "https://youtu.be/abc"
        assert data.current_song_title == "Test Song"
        assert data.current_song_duration == 240
        assert data.current_song_uploader == "Test Channel"
        assert data.current_song_requester_id == 333
        assert data.play_start_epoch == 1000.5
        assert data.total_pause_seconds == 12.5
        assert data.pause_start_epoch == 1100.0

    def test_empty_hash_yields_zero_value_snapshot(self) -> None:
        assert GuildStateData.from_redis({}) == GuildStateData()

    def test_partial_hash_missing_fields_get_defaults(self) -> None:
        data = GuildStateData.from_redis({b"volume": b"0.8"})
        assert data.volume == 0.8
        assert data.voice_channel_id is None
        assert data.current_song_url == ""
        assert data.total_pause_seconds == 0.0

    def test_malformed_float_yields_none_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"play_start_epoch": b"not-a-float"})
        assert data.play_start_epoch is None
        assert "play_start_epoch" in caplog.text

    def test_malformed_int_yields_none_and_warns(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"current_song_requester_id": b"abc"})
        assert data.current_song_requester_id is None
        assert "current_song_requester_id" in caplog.text

    def test_float_formatted_int_parses(self) -> None:
        data = GuildStateData.from_redis({b"voice_channel_id": b"111.0"})
        assert data.voice_channel_id == 111

    def test_snowflake_id_precision_preserved(self) -> None:
        # Discord snowflakes exceed float's 53-bit integer precision; parsing
        # via float() would corrupt 222222222222222222 to ...208.
        data = GuildStateData.from_redis(
            {b"current_song_requester_id": b"222222222222222222"}
        )
        assert data.current_song_requester_id == 222222222222222222

    def test_zero_volume_is_preserved(self) -> None:
        # Falsy-zero trap: coalescing with `or` would elevate a stored 0.0.
        data = GuildStateData.from_redis({b"volume": b"0.0"})
        assert data.volume == 0.0

    def test_missing_volume_is_none_not_default(self) -> None:
        # None means "nothing persisted" — callers skip the assignment instead
        # of clobbering live state with a fabricated 1.0 default.
        assert GuildStateData.from_redis({}).volume is None

    @pytest.mark.parametrize("raw", [b"nan", b"inf", b"-inf"])
    def test_non_finite_float_treated_as_malformed(
        self, raw: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        # nan/inf parse as floats but poison the position math downstream.
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"play_start_epoch": raw})
        assert data.play_start_epoch is None
        assert "play_start_epoch" in caplog.text

    def test_non_finite_int_field_treated_as_malformed(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        # int(float(b"inf")) raises OverflowError, not ValueError.
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"current_song_duration": b"inf"})
        assert data.current_song_duration is None

    def test_non_utf8_bytes_degrade_instead_of_raising(self) -> None:
        # A corrupt byte in one field must not make from_redis raise — that
        # would turn get_guild_state() into None ("Redis unavailable") and
        # block recovery entirely.
        data = GuildStateData.from_redis({b"current_song_title": b"Song \xff\xfe"})
        assert data.current_song_title.startswith("Song ")

    def test_empty_uploader_coerces_to_none(self) -> None:
        data = GuildStateData.from_redis({b"current_song_uploader": b""})
        assert data.current_song_uploader is None

    def test_empty_bytes_ids_coerce_to_none(self) -> None:
        data = GuildStateData.from_redis(
            {b"current_song_duration": b"", b"current_song_requester_id": b""}
        )
        assert data.current_song_duration is None
        assert data.current_song_requester_id is None

    def test_interjected_parses_one_as_true(self) -> None:
        data = GuildStateData.from_redis({b"current_song_interjected": b"1"})
        assert data.current_song_interjected is True

    @pytest.mark.parametrize("raw", [b"", b"0", b"true"])
    def test_interjected_anything_but_one_is_false(self, raw: Any) -> None:
        # The write path stores exactly "1" or "" — anything else (including
        # a missing field on pre-playnow state hashes) reads as False.
        data = GuildStateData.from_redis({b"current_song_interjected": raw})
        assert data.current_song_interjected is False

    def test_interjected_missing_is_false(self) -> None:
        assert GuildStateData.from_redis({}).current_song_interjected is False

    def test_enqueue_stamps_parse(self) -> None:
        data = GuildStateData.from_redis(
            {
                b"current_song_queued_at": b"1752530000.5",
                b"current_song_queue_position": b"3",
            }
        )
        assert data.current_song_queued_at == 1752530000.5
        assert data.current_song_queue_position == 3

    def test_enqueue_stamps_missing_are_zero(self) -> None:
        # Pre-feature state hashes carry neither field.
        data = GuildStateData.from_redis({})
        assert data.current_song_queued_at == 0.0
        assert data.current_song_queue_position == 0

    def test_query_source_parses(self) -> None:
        data = GuildStateData.from_redis(
            {b"current_song_query_source": b"soundcloud.com"}
        )
        assert data.current_song_query_source == "soundcloud.com"

    def test_query_source_missing_is_unknown(self) -> None:
        assert GuildStateData.from_redis({}).current_song_query_source == ""


class TestGuildStateDataProperties:
    def test_has_active_connection_true_with_both_ids(self) -> None:
        data = GuildStateData(voice_channel_id=1, text_channel_id=2)
        assert data.has_active_connection

    @pytest.mark.parametrize(
        "kwargs",
        [
            {},
            {"voice_channel_id": 1},
            {"text_channel_id": 2},
        ],
    )
    def test_has_active_connection_false_when_either_missing(self, kwargs: Any) -> None:
        assert not GuildStateData(**kwargs).has_active_connection

    def test_has_crashed_song(self) -> None:
        assert GuildStateData(current_song_url="https://x").has_crashed_song
        assert not GuildStateData().has_crashed_song

    def test_was_paused_at_crash(self) -> None:
        assert GuildStateData(pause_start_epoch=1.0).was_paused_at_crash
        assert not GuildStateData().was_paused_at_crash


class TestCrashedPositionAt:
    def test_none_without_play_start_epoch(self) -> None:
        assert GuildStateData().crashed_position_at(1000.0) is None

    def test_simple_elapsed(self) -> None:
        data = GuildStateData(play_start_epoch=1000.0)
        assert data.crashed_position_at(1042.0) == 42

    def test_accumulated_pause_subtracted(self) -> None:
        data = GuildStateData(play_start_epoch=1000.0, total_pause_seconds=10.0)
        assert data.crashed_position_at(1042.0) == 32

    def test_open_pause_interval_added_on_top(self) -> None:
        data = GuildStateData(
            play_start_epoch=1000.0,
            total_pause_seconds=10.0,
            pause_start_epoch=1030.0,
        )
        # elapsed 42 − (10 accumulated + 12 still-open) = 20
        assert data.crashed_position_at(1042.0) == 20

    def test_negative_result_clamped_to_zero(self) -> None:
        # Clock skew: play_start_epoch in the future relative to `now`.
        data = GuildStateData(play_start_epoch=2000.0)
        assert data.crashed_position_at(1000.0) == 0


class TestGuildStateDataImmutability:
    def test_frozen_assignment_raises(self) -> None:
        data = GuildStateData()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(data, "volume", 0.5)

    def test_slots_reject_unknown_attributes(self) -> None:
        data = GuildStateData()
        with pytest.raises((AttributeError, TypeError)):
            setattr(data, "unknown_attr", 1)


def _requester_stub(user_id: int, mention: str | None = None) -> discord.Member:
    """A stand-in for the requester Member: from_song/from_queue_object only read
    .id and .mention, and a real Member needs a live connection state.
    """
    return cast(
        discord.Member,
        SimpleNamespace(id=user_id, mention=mention or f"<@{user_id}>"),
    )


def _full_song_stub() -> YTDL:
    """A stand-in for a streamed song.

    YTDL subclasses FFmpegOpusAudio, so a real one spawns an FFmpeg subprocess;
    NowPlayingData.from_song only reads plain metadata attributes.
    """
    return cast(
        YTDL,
        SimpleNamespace(
            title="Test Song",
            webpage_url="https://youtu.be/abc",
            uploader="Test Channel",
            duration="4:00",
            duration_secs=240,
            thumbnail="https://img/x.jpg",
            views=1000,
            likes=50,
            abr=128,
            asr=48000,
            acodec="opus",
            requester=_requester_stub(333),
        ),
    )


def _empty_song_stub() -> YTDL:
    """As _full_song_stub, with every optional metadata field unset."""
    return cast(
        YTDL,
        SimpleNamespace(
            title=None,
            webpage_url=None,
            uploader=None,
            duration=None,
            duration_secs=0,
            thumbnail=None,
            views=None,
            likes=None,
            abr=None,
            asr=None,
            acodec=None,
            requester=None,
        ),
    )


class TestNowPlayingDataFromSong:
    def test_full_song(self) -> None:
        data = NowPlayingData.from_song(_full_song_stub())
        assert data.title == "Test Song"
        assert data.webpage_url == "https://youtu.be/abc"
        assert data.uploader == "Test Channel"
        assert data.duration == "4:00"
        assert data.thumbnail == "https://img/x.jpg"
        assert data.view_count == "1000"
        assert data.like_count == "50"
        assert data.abr == "128"
        assert data.asr == "48000"
        assert data.acodec == "opus"
        assert data.requester_id == "333"
        assert data.requester_mention == "<@333>"

    def test_all_none_optionals(self) -> None:
        data = NowPlayingData.from_song(_empty_song_stub())
        assert data.title == ""
        assert data.view_count == ""
        assert data.requester_id == ""
        assert data.requester_mention == "Unknown"

    def test_duration_blank_when_unknown(self) -> None:
        """A livestream has duration_secs == 0 but a non-empty duration string
        ("0:00"). Storing that would make the recovered embed claim a real
        length of zero; blank means the Duration line is omitted entirely,
        matching the live embed, which draws no bar in the same case."""
        song = cast(
            YTDL,
            SimpleNamespace(
                **{**vars(_full_song_stub()), "duration": "0:00", "duration_secs": 0},
            ),
        )
        assert NowPlayingData.from_song(song).duration == ""


class TestNowPlayingDataFromRedis:
    def test_empty_hash_returns_none(self) -> None:
        assert NowPlayingData.from_redis({}) is None

    def test_full_hash(self) -> None:
        raw = {
            k.encode(): v.encode()
            for k, v in NowPlayingData.from_song(_full_song_stub())
            .to_redis_mapping()
            .items()
        }
        data = NowPlayingData.from_redis(raw)
        assert data is not None
        assert data.title == "Test Song"
        assert data.requester_mention == "<@333>"

    def test_missing_requester_mention_defaults_to_unknown(self) -> None:
        data = NowPlayingData.from_redis({b"title": b"Test Song"})
        assert data is not None
        assert data.requester_mention == "Unknown"

    def test_round_trip_preserves_all_fields(self) -> None:
        original = NowPlayingData.from_song(_full_song_stub())
        # Encode step mirrors the wire format: to_redis_mapping() feeds HSET
        # mapping=, from_redis() consumes hgetall output (decode_responses=False).
        raw = {k.encode(): v.encode() for k, v in original.to_redis_mapping().items()}
        assert NowPlayingData.from_redis(raw) == original


class TestNowPlayingDataImmutability:
    def test_frozen_assignment_raises(self) -> None:
        data = NowPlayingData()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(data, "title", "x")


# ── Queue-entry value objects ─────────────────────────────────────────────────

# Golden wire fixtures — byte literals capturing the current writer output.
# These pin the wire format in both directions so a rolling restart can mix
# old and new writers. The _PRE_PLAYNOW golden pins the reader against entries
# written before the -playnow flags existed (parsed as False).

_PLAYNOW_FLAGS_FALSE = b'"interjected":false,"is_resume":false,"start_paused":false'
_ENQUEUE_STAMPS_ZERO = b'"queued_at":0.0,"queue_position":0'
_QUERY_SOURCE_UNKNOWN = b'"query_source":""'
_GOLDEN_QOBJ_FULL = (
    b'{"type":"qobj","webpage_url":"https://yt.com/v=1","title":"Golden Song","requester_id":222222222222222222,"ts":30,"user_input":"golden song","duration":240,"uploader":"Golden Channel","thumbnail":"https://img.yt/1.jpg","persisted":true,'
    + _PLAYNOW_FLAGS_FALSE
    + b","
    + _ENQUEUE_STAMPS_ZERO
    + b","
    + _QUERY_SOURCE_UNKNOWN
    + b"}"
)
_GOLDEN_QOBJ_BARE = (
    b'{"type":"qobj","webpage_url":"https://yt.com/v=2","title":"Bare","requester_id":42,"ts":null,"user_input":null,"duration":null,"uploader":null,"thumbnail":null,"persisted":true,'
    + _PLAYNOW_FLAGS_FALSE
    + b","
    + _ENQUEUE_STAMPS_ZERO
    + b","
    + _QUERY_SOURCE_UNKNOWN
    + b"}"
)
_GOLDEN_QOBJ_UNPERSISTED = (
    b'{"type":"qobj","webpage_url":"https://yt.com/v=4","title":"Crashed","requester_id":8,"ts":95,"user_input":null,"duration":180,"uploader":null,"thumbnail":null,"persisted":false,'
    + _PLAYNOW_FLAGS_FALSE
    + b","
    + _ENQUEUE_STAMPS_ZERO
    + b","
    + _QUERY_SOURCE_UNKNOWN
    + b"}"
)
_GOLDEN_QOBJ_PRE_PLAYNOW = b'{"type":"qobj","webpage_url":"https://yt.com/v=1","title":"Golden Song","requester_id":222222222222222222,"ts":30,"user_input":"golden song","duration":240,"uploader":"Golden Channel","thumbnail":"https://img.yt/1.jpg","persisted":true}'
_GOLDEN_YTSOURCE = (
    b'{"type":"ytsource","ytsearch":"ytsearch:some song","url":null,"process":true,"ts":null,'
    + _ENQUEUE_STAMPS_ZERO
    + b","
    + _QUERY_SOURCE_UNKNOWN
    + b"}"
)
# Written before the enqueue stamps existed: the reader defaults both to 0.
_GOLDEN_YTSOURCE_PRE_STAMPS = b'{"type":"ytsource","ytsearch":"ytsearch:some song","url":null,"process":true,"ts":null}'

_FULL_ENTRY = SongQueueEntry(
    webpage_url="https://yt.com/v=1",
    title="Golden Song",
    requester_id=222222222222222222,
    ts=30,
    user_input="golden song",
    duration=240,
    uploader="Golden Channel",
    thumbnail="https://img.yt/1.jpg",
)


class TestSongQueueEntryWire:
    def test_writer_matches_golden_bytes(self) -> None:
        assert _FULL_ENTRY.to_redis() == _GOLDEN_QOBJ_FULL

    def test_writer_matches_golden_bytes_nulls(self) -> None:
        entry = SongQueueEntry(
            webpage_url="https://yt.com/v=2", title="Bare", requester_id=42
        )
        assert entry.to_redis() == _GOLDEN_QOBJ_BARE

    def test_writer_matches_golden_bytes_unpersisted(self) -> None:
        entry = SongQueueEntry(
            webpage_url="https://yt.com/v=4",
            title="Crashed",
            requester_id=8,
            ts=95,
            duration=180,
            persisted=False,
        )
        assert entry.to_redis() == _GOLDEN_QOBJ_UNPERSISTED

    def test_reader_parses_golden_bytes(self) -> None:
        assert parse_queue_entry(_GOLDEN_QOBJ_FULL) == _FULL_ENTRY

    def test_reader_preserves_persisted_false(self) -> None:
        entry = parse_queue_entry(_GOLDEN_QOBJ_UNPERSISTED)
        assert isinstance(entry, SongQueueEntry)
        assert entry.persisted is False

    def test_round_trip(self) -> None:
        assert parse_queue_entry(_FULL_ENTRY.to_redis()) == _FULL_ENTRY

    def test_reader_parses_pre_playnow_entry_with_false_flags(self) -> None:
        # Entries written before the -playnow fields existed must parse with
        # all three flags defaulting False.
        assert parse_queue_entry(_GOLDEN_QOBJ_PRE_PLAYNOW) == _FULL_ENTRY

    def test_playnow_flags_round_trip(self) -> None:
        entry = dataclasses.replace(
            _FULL_ENTRY, interjected=True, is_resume=True, start_paused=True
        )
        parsed = parse_queue_entry(entry.to_redis())
        assert parsed == entry
        assert isinstance(parsed, SongQueueEntry)
        assert (parsed.interjected, parsed.is_resume, parsed.start_paused) == (
            True,
            True,
            True,
        )

    def test_snowflake_requester_id_exact(self) -> None:
        entry = parse_queue_entry(_GOLDEN_QOBJ_FULL)
        assert isinstance(entry, SongQueueEntry)
        assert entry.requester_id == 222222222222222222  # no float path

    def test_from_queue_object(self) -> None:
        item = QueueObject(
            webpage_url="https://yt.com/v=1",
            title="Golden Song",
            requester=_requester_stub(222222222222222222),
            ts=30,
            user_input="golden song",
            duration=240,
            uploader="Golden Channel",
            thumbnail="https://img.yt/1.jpg",
            persisted=True,
            interjected=False,
            is_resume=False,
            start_paused=False,
        )
        assert SongQueueEntry.from_queue_object(item) == _FULL_ENTRY

    def test_from_queue_object_carries_playnow_flags(self) -> None:
        item = QueueObject(
            webpage_url="https://yt.com/v=1",
            title="Golden Song",
            requester=_requester_stub(222222222222222222),
            ts=151,
            user_input=None,
            duration=240,
            uploader="Golden Channel",
            thumbnail=None,
            persisted=True,
            interjected=False,
            is_resume=True,
            start_paused=True,
        )
        entry = SongQueueEntry.from_queue_object(item)
        assert entry.is_resume is True
        assert entry.start_paused is True
        assert entry.interjected is False

    def test_reader_parses_pre_stamp_entry_with_zero_stamps(self) -> None:
        # Entries written before the enqueue stamps existed default both to 0.
        entry = parse_queue_entry(_GOLDEN_QOBJ_PRE_PLAYNOW)
        assert isinstance(entry, SongQueueEntry)
        assert (entry.queued_at, entry.queue_position) == (0.0, 0)

    def test_enqueue_stamps_round_trip(self) -> None:
        item = QueueObject(
            webpage_url="https://yt.com/v=1",
            title="Golden Song",
            requester=_requester_stub(222222222222222222),
            queued_at=1752530000.5,
            queue_position=4,
        )
        entry = SongQueueEntry.from_queue_object(item)
        parsed = parse_queue_entry(entry.to_redis())
        assert parsed == entry
        assert isinstance(parsed, SongQueueEntry)
        assert (parsed.queued_at, parsed.queue_position) == (1752530000.5, 4)

    def test_query_source_round_trips(self) -> None:
        item = QueueObject(
            webpage_url="https://yt.com/v=1",
            title="Golden Song",
            requester=_requester_stub(222222222222222222),
            query_source="tiktok.com",
        )
        parsed = parse_queue_entry(SongQueueEntry.from_queue_object(item).to_redis())
        assert isinstance(parsed, SongQueueEntry)
        assert parsed.query_source == "tiktok.com"

    def test_reader_defaults_query_source_on_a_pre_feature_entry(self) -> None:
        entry = parse_queue_entry(_GOLDEN_QOBJ_PRE_PLAYNOW)
        assert isinstance(entry, SongQueueEntry)
        assert entry.query_source == ""


class TestSearchQueueEntryWire:
    def test_writer_matches_golden_bytes(self) -> None:
        entry = SearchQueueEntry(ytsearch="ytsearch:some song", process=True)
        assert entry.to_redis() == _GOLDEN_YTSOURCE

    def test_reader_parses_golden_bytes(self) -> None:
        entry = parse_queue_entry(_GOLDEN_YTSOURCE)
        assert entry == SearchQueueEntry(ytsearch="ytsearch:some song", process=True)

    def test_round_trip(self) -> None:
        entry = SearchQueueEntry(url="https://yt.com/v=9", ts=10)
        assert parse_queue_entry(entry.to_redis()) == entry

    def test_from_ytsource(self) -> None:
        source = YTSource(ytsearch="ytsearch:x", url=None, process=True, ts=None)
        entry = SearchQueueEntry.from_ytsource(source)
        assert entry == SearchQueueEntry(ytsearch="ytsearch:x", process=True)

    def test_reader_parses_pre_stamp_entry_with_zero_stamps(self) -> None:
        # Searches written before the enqueue stamps existed must still parse.
        entry = parse_queue_entry(_GOLDEN_YTSOURCE_PRE_STAMPS)
        assert isinstance(entry, SearchQueueEntry)
        assert (entry.queued_at, entry.queue_position) == (0.0, 0)
        assert entry.query_source == ""

    def test_query_source_round_trips_through_the_lazy_resolve(self) -> None:
        # The leg that makes a Spotify playlist track archive as Spotify: these
        # entries sit in Redis unresolved and become YouTube URLs at dequeue, so
        # nothing downstream could recover the classification.
        source = YTSource(
            ytsearch="ytsearch:x", process=True, query_source="spotify.com"
        )
        entry = SearchQueueEntry.from_ytsource(source)
        parsed = parse_queue_entry(entry.to_redis())
        assert isinstance(parsed, SearchQueueEntry)
        assert parsed.query_source == "spotify.com"

    def test_enqueue_stamps_round_trip(self) -> None:
        source = YTSource(
            ytsearch="ytsearch:x", queued_at=1752530000.5, queue_position=7
        )
        entry = SearchQueueEntry.from_ytsource(source)
        parsed = parse_queue_entry(entry.to_redis())
        assert parsed == entry
        assert isinstance(parsed, SearchQueueEntry)
        assert (parsed.queued_at, parsed.queue_position) == (1752530000.5, 7)


class TestParseQueueEntryCorrupt:
    @pytest.mark.parametrize(
        "raw",
        [
            b"not json at all",
            b'{"type":"qobj","title":"missing url and requester"}',
            b"",
        ],
    )
    def test_corrupt_entry_dropped_with_warning(
        self, raw: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            assert parse_queue_entry(raw) is None
        assert "corrupt queue entry" in caplog.text


def _history_entry(**overrides: Any) -> HistoryEntry:
    fields: dict = dict(
        # Real snowflake magnitude: must survive int() parsing without float
        # precision loss (same hazard as requester_id snowflakes).
        guild_id=111111111111111111,
        title="Song Title",
        webpage_url="https://yt.com/v=1",
        duration_secs=242,
        played_secs=225,
        requester_id=42,
        requester_name="Omkar",
        thumbnail="https://i.ytimg.com/t.jpg",
        uploader="Chan",
        played_at=1752530000.0,
        message_id=999999999999999999,
        queued_at=1752529000.0,
        queue_position=2,
        query_source="youtube.com",
    )
    fields.update(overrides)
    return HistoryEntry(**fields)


class TestHistoryEntryWire:
    def test_golden_bytes(self) -> None:
        # Wire format pinned: rolling restarts mix writers, so the field
        # names and value encodings must not drift.
        assert serialize_history_entry(_history_entry()) == (
            b'{"guild_id":111111111111111111,"title":"Song Title",'
            b'"webpage_url":"https://yt.com/v=1",'
            b'"duration_secs":242,"played_secs":225,"requester_id":42,'
            b'"requester_name":"Omkar","thumbnail":"https://i.ytimg.com/t.jpg",'
            b'"uploader":"Chan","played_at":1752530000.0,'
            b'"message_id":999999999999999999,'
            b'"queued_at":1752529000.0,"queue_position":2,'
            b'"query_source":"youtube.com"}'
        )

    def test_round_trip(self) -> None:
        entry = _history_entry()
        assert parse_history_entry(serialize_history_entry(entry)) == entry

    def test_pre_postgres_entry_parses_with_guild_id_zero(self) -> None:
        # Golden bytes from the pre-guild_id writer (history overhaul era) —
        # at-rest entries mix writers, so these must stay readable, with the
        # absent fields defaulting to 0 (backfill stamps the real guild id from
        # the key; message_id has no such recovery and stays unknown).
        pre_postgres = (
            b'{"title":"Song Title","webpage_url":"https://yt.com/v=1",'
            b'"duration_secs":242,"played_secs":225,"requester_id":42,'
            b'"requester_name":"Omkar","thumbnail":"https://i.ytimg.com/t.jpg",'
            b'"uploader":"Chan","played_at":1752530000.0}'
        )
        assert parse_history_entry(pre_postgres) == _history_entry(
            guild_id=0,
            message_id=0,
            queued_at=0.0,
            queue_position=0,
            query_source="",
        )

    def test_pre_message_id_entry_parses_with_message_id_zero(self) -> None:
        # The narrower mixed-build case, and the one a rolling restart actually
        # produces: a writer that already stamped guild_id but predates
        # message_id. Its entries must keep parsing, not drop as corrupt.
        pre_message_id = (
            b'{"guild_id":111111111111111111,"title":"Song Title",'
            b'"webpage_url":"https://yt.com/v=1",'
            b'"duration_secs":242,"played_secs":225,"requester_id":42,'
            b'"requester_name":"Omkar","thumbnail":"https://i.ytimg.com/t.jpg",'
            b'"uploader":"Chan","played_at":1752530000.0}'
        )
        assert parse_history_entry(pre_message_id) == _history_entry(
            message_id=0, queued_at=0.0, queue_position=0, query_source=""
        )

    def test_pre_enqueue_stamp_entry_parses_with_zero_stamps(self) -> None:
        # The newest mixed-build case: a writer that predates queued_at /
        # queue_position. Both default to the "unknown" zero value.
        pre_stamps = (
            b'{"guild_id":111111111111111111,"title":"Song Title",'
            b'"webpage_url":"https://yt.com/v=1",'
            b'"duration_secs":242,"played_secs":225,"requester_id":42,'
            b'"requester_name":"Omkar","thumbnail":"https://i.ytimg.com/t.jpg",'
            b'"uploader":"Chan","played_at":1752530000.0,'
            b'"message_id":999999999999999999}'
        )
        assert parse_history_entry(pre_stamps) == _history_entry(
            queued_at=0.0, queue_position=0, query_source=""
        )

    def test_snowflake_guild_id_survives_round_trip(self) -> None:
        entry = _history_entry(guild_id=222222222222222222)
        parsed = parse_history_entry(serialize_history_entry(entry))
        assert parsed is not None
        assert parsed.guild_id == 222222222222222222

    def test_snowflake_message_id_survives_round_trip(self) -> None:
        # Same float-precision hazard as the other snowflake columns: a
        # message id routed through a float would come back off by a digit.
        entry = _history_entry(message_id=333333333333333333)
        parsed = parse_history_entry(serialize_history_entry(entry))
        assert parsed is not None
        assert parsed.message_id == 333333333333333333

    def test_unknown_keys_ignored_and_missing_keys_default(self) -> None:
        # Forward/backward tolerance: a newer writer's extra field must not
        # break this reader, and absent fields become zero-values.
        raw = orjson.dumps({"title": "x", "future_field": 1})
        assert parse_history_entry(raw) == HistoryEntry(title="x")

    @pytest.mark.parametrize(
        "raw",
        [
            b"not json at all",
            b"123",
            b"[1, 2]",
            b"",
            # Pre-overhaul "<title> - <url>" strings: the deployment storage
            # was recreated, so these no longer parse — they drop as corrupt.
            b'"Old Song - https://yt.com/v=old"',
            b'{"title": "x", "duration_secs": "not a number"}',
            b'{"title": "x", "played_at": {"nested": true}}',
        ],
    )
    def test_corrupt_entry_dropped_with_warning(
        self, raw: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            assert parse_history_entry(raw) is None
        assert "corrupt history entry" in caplog.text


def _entry_with(**kwargs: Any) -> HistoryEntry:
    """Constructor shim for the parametrized domain tests below.

    Splatting a parametrized field name into HistoryEntry makes pyright check the
    value against every field's type; widening here confines the Any to one line.
    """
    return HistoryEntry(**kwargs)


class TestHistoryEntryDomain:
    """The schema lock: constructing a HistoryEntry is the proof that Postgres
    accepts it, so every clamp is asserted on the TYPE rather than on any archive
    helper — the property is about the value object, not the repository.
    """

    def test_clean_entry_is_untouched(self) -> None:
        entry = HistoryEntry(
            guild_id=42,
            title="Song",
            webpage_url="https://yt.com/v=1",
            duration_secs=200,
            played_secs=200,
            requester_id=222222222222222222,
            requester_name="user",
            thumbnail="https://img/x.jpg",
            uploader="chan",
            played_at=1000.0,
            message_id=333333333333333333,
            queued_at=900.0,
            queue_position=3,
            query_source="spotify.com",
        )
        assert dataclasses.asdict(entry) == {
            "guild_id": 42,
            "title": "Song",
            "webpage_url": "https://yt.com/v=1",
            "duration_secs": 200,
            "played_secs": 200,
            "requester_id": 222222222222222222,
            "requester_name": "user",
            "thumbnail": "https://img/x.jpg",
            "uploader": "chan",
            "played_at": 1000.0,
            "message_id": 333333333333333333,
            "queued_at": 900.0,
            "queue_position": 3,
            "query_source": "spotify.com",
        }

    def test_nul_stripped_from_every_text_field(self) -> None:
        # Postgres text cannot hold NUL; the server raises
        # CharacterNotInRepertoireError and takes the whole batch with it.
        entry = HistoryEntry(
            guild_id=42,
            title="ab\x00cd",
            webpage_url="https://yt\x00.com",
            requester_name="u\x00ser",
            thumbnail="\x00",
            uploader="chan\x00",
        )
        assert entry.title == "abcd"
        assert entry.webpage_url == "https://yt.com"
        assert entry.requester_name == "user"
        assert entry.thumbnail == ""
        assert entry.uploader == "chan"

    def test_nul_strip_leaves_other_fields_alone(self) -> None:
        entry = HistoryEntry(guild_id=42, title="a\x00b", uploader="chan")
        assert entry.title == "ab"
        assert entry.uploader == "chan"
        assert entry.guild_id == 42

    @pytest.mark.parametrize(
        "played_at",
        [1e18, 1e300, -1e12, -1e18, float("nan"), float("inf"), float("-inf")],
        ids=["huge", "absurd", "negative", "very-negative", "nan", "inf", "-inf"],
    )
    def test_out_of_range_epoch_collapses_to_unknown_sentinel(
        self, played_at: float
    ) -> None:
        # datetime.fromtimestamp would raise in _entry_to_row, before any SQL
        # is sent. The chained comparison is False for NaN, which is why it is
        # written as a range test rather than two explicit bounds checks.
        assert HistoryEntry(guild_id=1, played_at=played_at).played_at == 0.0

    @pytest.mark.parametrize(
        "queued_at",
        [1e18, -1e12, float("nan"), float("inf")],
        ids=["huge", "negative", "nan", "inf"],
    )
    def test_out_of_range_queued_at_collapses_to_unknown_sentinel(
        self, queued_at: float
    ) -> None:
        # queued_at rides _EPOCH_FIELDS, so it clamps exactly like played_at.
        assert HistoryEntry(guild_id=1, queued_at=queued_at).queued_at == 0.0

    def test_negative_queue_position_clamps_to_zero(self) -> None:
        # int4 domain, and play_history's CHECK is >= 0.
        assert HistoryEntry(guild_id=1, queue_position=-3).queue_position == 0

    def test_timestamptz_ceiling_is_accepted_unchanged(self) -> None:
        # The bound is inclusive: 9999-12-31T23:59:59Z is a legal timestamptz.
        assert HistoryEntry(guild_id=1, played_at=253402300799.0).played_at == (
            253402300799.0
        )
        assert HistoryEntry(guild_id=1, queued_at=253402300799.0).queued_at == (
            253402300799.0
        )

    @pytest.mark.parametrize("field", ["duration_secs", "played_secs"])
    def test_int4_columns_clamp_at_the_int4_ceiling(self, field: str) -> None:
        entry = _entry_with(guild_id=1, **{field: 2**31})
        assert getattr(entry, field) == 2**31 - 1

    @pytest.mark.parametrize("field", ["guild_id", "requester_id", "message_id"])
    @pytest.mark.parametrize("value", [2**63, 2**64 - 1], ids=["int8-max+1", "uint64"])
    def test_int8_columns_clamp_at_the_signed_bigint_ceiling(
        self, field: str, value: int
    ) -> None:
        # The gap the earlier proposal missed: orjson happily encodes anything
        # up to 2**64-1, but the column is a SIGNED bigint. Both of these are
        # wire-legal and would be refused by Postgres.
        entry = _entry_with(**{field: value})
        assert getattr(entry, field) == 2**63 - 1

    @pytest.mark.parametrize(
        "field",
        ["guild_id", "requester_id", "message_id", "duration_secs", "played_secs"],
    )
    def test_negative_ints_clamp_to_zero(self, field: str) -> None:
        assert getattr(_entry_with(**{field: -5}), field) == 0

    def test_snowflake_ids_survive(self) -> None:
        # int8 columns: a real Discord snowflake must not be clamped into
        # nonsense by an over-eager ceiling.
        assert HistoryEntry(requester_id=222222222222222222).requester_id == (
            222222222222222222
        )
        assert HistoryEntry(message_id=333333333333333333).message_id == (
            333333333333333333
        )

    def test_replace_re_runs_the_validator(self) -> None:
        # backfill_history stamps guild_id onto legacy entries
        # with dataclasses.replace, on the least trustworthy data in the system.
        # If that bypassed __post_init__ the whole lock would be decorative.
        entry = HistoryEntry(guild_id=1, title="x")
        assert dataclasses.replace(entry, guild_id=2**64 - 1).guild_id == 2**63 - 1

    def test_every_field_is_covered_by_a_domain(self) -> None:
        """The lock is only closed if every field is in it. A field added to the
        dataclass but not to a domain tuple is silently unvalidated — and would
        be found by a production CHECK violation instead of by this test."""
        from src.guild_state import (
            _EPOCH_FIELDS,
            _INT4_FIELDS,
            _INT8_FIELDS,
            _SLUG_FIELDS,
            _TEXT_FIELDS,
        )

        covered = {
            *_TEXT_FIELDS,
            *_INT4_FIELDS,
            *_INT8_FIELDS,
            *_EPOCH_FIELDS,
            *_SLUG_FIELDS,
        }
        assert {f.name for f in dataclasses.fields(HistoryEntry)} == covered

    @pytest.mark.parametrize(
        "value,expected",
        [
            ("search", "search"),
            ("youtube.com", "youtube.com"),
            ("artist.bandcamp.com", "artist.bandcamp.com"),
            ("", ""),
            # Every producer defect clamps to the unknown sentinel rather than
            # reaching the column, which is what keeps 0001's CHECK unfireable.
            ("Spotify.COM", ""),
            ("a" * 65, ""),
            ("evil.com](http://x)", ""),
            ("nul\x00.com", ""),
            ("two words", ""),
        ],
    )
    def test_query_source_is_clamped_to_the_column_domain(
        self, value: str, expected: str
    ) -> None:
        assert HistoryEntry(guild_id=1, query_source=value).query_source == expected

    def test_query_source_rejects_a_non_string_like_every_text_field(self) -> None:
        """__post_init__ coerces nothing — storing a row reading "None" would be
        worse than the TypeError. Matches the documented behaviour of title."""
        with pytest.raises(TypeError):
            HistoryEntry(guild_id=1, query_source=cast(str, object()))

    def test_every_clamped_entry_survives_the_wire(self) -> None:
        # End-to-end: anything __post_init__ produces must also serialize and
        # parse back unchanged, since the outbox is the path to Postgres.
        for entry in (
            HistoryEntry(guild_id=1, title="x\x00y"),
            HistoryEntry(guild_id=2**64 - 1, played_at=1e18),
            HistoryEntry(guild_id=1, duration_secs=-5, played_at=float("nan")),
        ):
            assert parse_history_entry(serialize_history_entry(entry)) == entry


def test_orjson_refuses_what_post_init_cannot_clamp() -> None:
    """The wire codec is the first half of the validator, and it is documented
    nowhere a reader would find it. A lone surrogate cannot be encoded, so it
    can never reach Redis — which is why __post_init__ has no surrogate arm and
    why the residual poison set is only four vectors wide."""
    with pytest.raises(TypeError):
        serialize_history_entry(HistoryEntry(guild_id=1, title="a\ud800b"))


def _history_song_stub(**overrides: Any) -> YTDL:
    fields: dict = dict(
        title="Test Song",
        webpage_url="https://youtu.be/abc",
        uploader="Test Channel",
        duration_secs=242,
        position_secs=225.0,
        thumbnail="https://img/x.jpg",
        requester=SimpleNamespace(id=333, display_name="Omkar"),
        queued_at=1752529000.0,
        queue_position=2,
        query_source="youtube.com",
    )
    fields.update(overrides)
    return cast(YTDL, SimpleNamespace(**fields))


class TestHistoryEntryFromSong:
    def test_maps_song_fields(self) -> None:
        entry = HistoryEntry.from_song(
            _history_song_stub(),
            guild_id=111,
            played_at=1752530000.0,
            message_id=444444444444444444,
        )
        assert entry == HistoryEntry(
            guild_id=111,
            title="Test Song",
            webpage_url="https://youtu.be/abc",
            duration_secs=242,
            played_secs=225,
            requester_id=333,
            requester_name="Omkar",
            thumbnail="https://img/x.jpg",
            uploader="Test Channel",
            played_at=1752530000.0,
            message_id=444444444444444444,
            queued_at=1752529000.0,
            queue_position=2,
            query_source="youtube.com",
        )

    def test_guild_id_and_message_id_are_required_keywords(self) -> None:
        # Requiredness is the whole protection for message_id: play_history's
        # CHECK is `message_id >= 0` with default 0, so a forgotten stamp inserts
        # cleanly and reads as a song that genuinely had no host. pyright catches
        # a missing argument only while no default exists — adding one silently
        # retires the guarantee, and this is what notices.
        song = _history_song_stub()
        with pytest.raises(TypeError):
            HistoryEntry.from_song(song, guild_id=111, played_at=1.0)  # pyright: ignore[reportCallIssue]
        with pytest.raises(TypeError):
            HistoryEntry.from_song(song, played_at=1.0, message_id=0)  # pyright: ignore[reportCallIssue]

    def test_message_id_is_caller_supplied(self) -> None:
        # The song object carries no message id — it comes from the NP host the
        # playback loop captured at song end, and 0 means "no message hosted this
        # song's block" rather than "the caller forgot". Both arms are needed: 0
        # is also the dataclass default, so asserting it alone passes even if
        # from_song hardcodes 0, drops the argument, or never sets the field. The
        # non-zero arm is what makes the zero arm mean "0 because the caller said".
        song = _history_song_stub()
        base = {"guild_id": 111, "played_at": 1.0}
        assert HistoryEntry.from_song(song, **base, message_id=0).message_id == 0
        assert HistoryEntry.from_song(song, **base, message_id=888).message_id == 888

    def test_played_secs_is_position_reached(self) -> None:
        song = _history_song_stub(position_secs=100.4)
        assert (
            HistoryEntry.from_song(
                song, guild_id=111, played_at=1.0, message_id=0
            ).played_secs
            == 100
        )

    def test_played_secs_capped_at_duration(self) -> None:
        # position can exceed duration by fractions of a frame at natural end.
        song = _history_song_stub(position_secs=243.02)
        assert (
            HistoryEntry.from_song(
                song, guild_id=111, played_at=1.0, message_id=0
            ).played_secs
            == 242
        )

    def test_unknown_duration_leaves_position_uncapped(self) -> None:
        song = _history_song_stub(duration_secs=0, position_secs=99.6)
        entry = HistoryEntry.from_song(song, guild_id=111, played_at=1.0, message_id=0)
        assert entry.duration_secs == 0
        assert entry.played_secs == 100

    def test_no_requester_degrades_to_zero_values(self) -> None:
        song = _history_song_stub(requester=None)
        entry = HistoryEntry.from_song(song, guild_id=111, played_at=1.0, message_id=0)
        assert entry.requester_id == 0
        assert entry.requester_name == ""

    def test_none_metadata_degrades_to_zero_values(self) -> None:
        # yt-dlp can return None for any metadata field.
        song = _history_song_stub(
            title=None, webpage_url=None, uploader=None, thumbnail=None
        )
        entry = HistoryEntry.from_song(song, guild_id=111, played_at=1.0, message_id=0)
        assert entry.title == ""
        assert entry.webpage_url == ""
        assert entry.uploader == ""
        assert entry.thumbnail == ""


class TestFromCrashedState:
    def test_none_when_no_crashed_song(self) -> None:
        assert SongQueueEntry.from_crashed_state(GuildStateData(), position=10) is None

    def test_maps_crashed_fields(self) -> None:
        state = GuildStateData(
            current_song_url="https://yt.com/v=crash",
            current_song_title="Crashed",
            current_song_duration=180,
            current_song_uploader="Chan",
            current_song_requester_id=42,
        )
        entry = SongQueueEntry.from_crashed_state(state, position=95)
        assert entry == SongQueueEntry(
            webpage_url="https://yt.com/v=crash",
            title="Crashed",
            requester_id=42,
            ts=95,
            duration=180,
            uploader="Chan",
            persisted=False,
        )

    def test_persisted_false_and_position_none_passthrough(self) -> None:
        state = GuildStateData(current_song_url="https://x", current_song_title="T")
        entry = SongQueueEntry.from_crashed_state(state, position=None)
        assert entry is not None
        assert entry.persisted is False
        assert entry.ts is None
        assert entry.requester_id is None  # no requester recorded

    def test_interjected_flag_survives_crash(self) -> None:
        # A crash mid-interjection must not demote the recovered song: a
        # -playnow after recovery still replaces it instead of stacking.
        state = GuildStateData(
            current_song_url="https://x",
            current_song_title="T",
            current_song_interjected=True,
        )
        entry = SongQueueEntry.from_crashed_state(state, position=42)
        assert entry is not None
        assert entry.interjected is True

    def test_interjected_defaults_false(self) -> None:
        state = GuildStateData(current_song_url="https://x", current_song_title="T")
        entry = SongQueueEntry.from_crashed_state(state, position=None)
        assert entry is not None
        assert entry.interjected is False

    def test_enqueue_stamps_survive_crash(self) -> None:
        # Recovery re-queues the song but does not re-queue the play: it keeps
        # the position it was originally given, so the archive stays truthful.
        state = GuildStateData(
            current_song_url="https://x",
            current_song_title="T",
            current_song_queued_at=1752530000.5,
            current_song_queue_position=3,
        )
        entry = SongQueueEntry.from_crashed_state(state, position=42)
        assert entry is not None
        assert (entry.queued_at, entry.queue_position) == (1752530000.5, 3)

    def test_enqueue_stamps_default_to_unknown(self) -> None:
        state = GuildStateData(current_song_url="https://x", current_song_title="T")
        entry = SongQueueEntry.from_crashed_state(state, position=None)
        assert entry is not None
        assert (entry.queued_at, entry.queue_position) == (0.0, 0)

    def test_query_source_survives_crash(self) -> None:
        # The classification is only knowable at parse time, so a crash is the one
        # place it can be lost — and it would be lost silently, and only for
        # crashed plays.
        state = GuildStateData(
            current_song_url="https://x",
            current_song_title="T",
            current_song_query_source="spotify.com",
        )
        entry = SongQueueEntry.from_crashed_state(state, position=42)
        assert entry is not None
        assert entry.query_source == "spotify.com"


class TestGuildPlaybackSnapshot:
    @pytest.mark.parametrize(
        "queue,crashed,expected",
        [
            ((), False, False),
            ((), True, True),
            (("entry",), False, True),
            (("entry",), True, True),
        ],
    )
    def test_has_restorable_playback_truth_table(
        self, queue: Any, crashed: Any, expected: Any
    ) -> None:
        state = GuildStateData(current_song_url="https://x" if crashed else "")
        entries = tuple(
            SongQueueEntry(webpage_url="https://q", title="Q", requester_id=1)
            for _ in queue
        )
        snap = GuildPlaybackSnapshot(state=state, queue=entries)
        assert snap.has_restorable_playback is expected

    def test_pending_count(self) -> None:
        entry = SongQueueEntry(webpage_url="https://q", title="Q", requester_id=1)
        assert GuildPlaybackSnapshot(state=GuildStateData()).pending_count == 0
        assert (
            GuildPlaybackSnapshot(
                state=GuildStateData(), queue=(entry, entry)
            ).pending_count
            == 2
        )

    def test_frozen(self) -> None:
        snap = GuildPlaybackSnapshot(state=GuildStateData())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(snap, "queue", ())


class TestGuildRecoveryGate:
    @pytest.mark.parametrize(
        "pending_count,crashed,expected",
        [
            (0, False, False),
            (0, True, True),
            (2, False, True),
            (2, True, True),
        ],
    )
    def test_has_restorable_playback_truth_table(
        self, pending_count: int, crashed: Any, expected: Any
    ) -> None:
        """Mirrors GuildPlaybackSnapshot's gate, over the queue length instead
        of the queue tuple."""
        state = GuildStateData(current_song_url="https://x" if crashed else "")
        gate = GuildRecoveryGate(state=state, pending_count=pending_count)
        assert gate.has_restorable_playback is expected

    def test_frozen(self) -> None:
        gate = GuildRecoveryGate(state=GuildStateData())
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(gate, "pending_count", 5)


class TestQueueEntryImmutability:
    def test_song_entry_frozen(self) -> None:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(_FULL_ENTRY, "title", "x")

    def test_search_entry_frozen(self) -> None:
        entry = SearchQueueEntry()
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(entry, "url", "x")


class TestGuildConfig:
    """Durable per-guild preferences. The tri-state is the whole design: "never
    chose" has to stay distinguishable from "chose off", or a guild that opted out
    while DEBUG_MODE=true would silently opt back in."""

    def test_nothing_stored_is_unset_not_off(self) -> None:
        assert GuildConfig.from_redis({}).debug_mode is None

    @pytest.mark.parametrize(("stored", "expected"), [(b"1", True), (b"0", False)])
    def test_a_stored_choice_round_trips(self, stored: bytes, expected: bool) -> None:
        raw = {ConfigField.DEBUG_MODE.encode(): stored}
        config = GuildConfig.from_redis(raw)
        assert config.debug_mode is expected
        assert GuildConfig(debug_mode=expected).to_redis() == {
            ConfigField.DEBUG_MODE: stored.decode()
        }

    def test_unset_writes_no_field_at_all(self) -> None:
        """Absent, not a sentinel — otherwise the read path cannot tell the two
        apart and the tri-state collapses on the first round trip."""
        assert GuildConfig().to_redis() == {}

    @pytest.mark.parametrize("garbage", [b"", b"true", b"yes", b"2", b"\xff"])
    def test_an_unparseable_value_reads_as_unset(self, garbage: bytes) -> None:
        """This key outlives builds. One field written by a future version must cost
        the guild its default, not its whole config — the same rule the queue and
        history parsers follow."""
        raw = {ConfigField.DEBUG_MODE.encode(): garbage}
        assert GuildConfig.from_redis(raw).debug_mode is None


class TestStoredVolumeMigration:
    """Volume moved from guild:{id}:state to guild:{id}:config. The fallback is a
    one-release migration path, not a permanent dual-read: dropping the legacy field
    outright would have silently reset every deployed guild to 100%."""

    def test_config_wins_when_both_are_present(self) -> None:
        snapshot = GuildPlaybackSnapshot(
            state=GuildStateData(volume=0.10),
            config=GuildConfig(volume=0.75),
        )
        assert snapshot.stored_volume == 0.75

    def test_a_legacy_value_is_still_honoured(self) -> None:
        """The guild set 30% before the deploy; it must come back at 30%."""
        snapshot = GuildPlaybackSnapshot(state=GuildStateData(volume=0.30))
        assert snapshot.stored_volume == 0.30

    def test_nothing_stored_stays_none(self) -> None:
        """Not 1.0 — restore has to skip the assignment rather than clobber a
        concurrent -volume with a fabricated default."""
        snapshot = GuildPlaybackSnapshot(state=GuildStateData())
        assert snapshot.stored_volume is None

    def test_an_explicit_zero_is_not_mistaken_for_unset(self) -> None:
        """0.0 is falsy and a real choice — muted."""
        snapshot = GuildPlaybackSnapshot(
            state=GuildStateData(), config=GuildConfig(volume=0.0)
        )
        assert snapshot.stored_volume == 0.0
