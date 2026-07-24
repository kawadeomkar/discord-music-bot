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
    """A stand-in for the requester Member.

    from_song/from_queue_object only ever read .id and .mention off it, and a
    real Member needs a live connection state to construct.
    """
    return cast(
        discord.Member,
        SimpleNamespace(id=user_id, mention=mention or f"<@{user_id}>"),
    )


def _full_song_stub() -> YTDL:
    """A stand-in for a streamed song.

    YTDL subclasses FFmpegOpusAudio, so building a real one spawns an FFmpeg
    subprocess. NowPlayingData.from_song only reads plain metadata attributes,
    so a namespace carrying them is a faithful substitute.
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
_GOLDEN_QOBJ_FULL = (
    b'{"type":"qobj","webpage_url":"https://yt.com/v=1","title":"Golden Song","requester_id":222222222222222222,"ts":30,"user_input":"golden song","duration":240,"uploader":"Golden Channel","thumbnail":"https://img.yt/1.jpg","persisted":true,'
    + _PLAYNOW_FLAGS_FALSE
    + b"}"
)
_GOLDEN_QOBJ_BARE = (
    b'{"type":"qobj","webpage_url":"https://yt.com/v=2","title":"Bare","requester_id":42,"ts":null,"user_input":null,"duration":null,"uploader":null,"thumbnail":null,"persisted":true,'
    + _PLAYNOW_FLAGS_FALSE
    + b"}"
)
_GOLDEN_QOBJ_UNPERSISTED = (
    b'{"type":"qobj","webpage_url":"https://yt.com/v=4","title":"Crashed","requester_id":8,"ts":95,"user_input":null,"duration":180,"uploader":null,"thumbnail":null,"persisted":false,'
    + _PLAYNOW_FLAGS_FALSE
    + b"}"
)
_GOLDEN_QOBJ_PRE_PLAYNOW = b'{"type":"qobj","webpage_url":"https://yt.com/v=1","title":"Golden Song","requester_id":222222222222222222,"ts":30,"user_input":"golden song","duration":240,"uploader":"Golden Channel","thumbnail":"https://img.yt/1.jpg","persisted":true}'
_GOLDEN_YTSOURCE = b'{"type":"ytsource","ytsearch":"ytsearch:some song","url":null,"process":true,"ts":null}'

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
            b'"uploader":"Chan","played_at":1752530000.0}'
        )

    def test_round_trip(self) -> None:
        entry = _history_entry()
        assert parse_history_entry(serialize_history_entry(entry)) == entry

    def test_pre_postgres_entry_parses_with_guild_id_zero(self) -> None:
        # Golden bytes from the pre-guild_id writer (history overhaul era) —
        # at-rest entries mix writers, so these must stay readable, with the
        # absent field defaulting to 0 (backfill stamps the real id from the
        # key — docs/POSTGRES_HISTORY_PLAN.md §5.6).
        pre_postgres = (
            b'{"title":"Song Title","webpage_url":"https://yt.com/v=1",'
            b'"duration_secs":242,"played_secs":225,"requester_id":42,'
            b'"requester_name":"Omkar","thumbnail":"https://i.ytimg.com/t.jpg",'
            b'"uploader":"Chan","played_at":1752530000.0}'
        )
        assert parse_history_entry(pre_postgres) == _history_entry(guild_id=0)

    def test_snowflake_guild_id_survives_round_trip(self) -> None:
        entry = _history_entry(guild_id=222222222222222222)
        parsed = parse_history_entry(serialize_history_entry(entry))
        assert parsed is not None
        assert parsed.guild_id == 222222222222222222

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


def _history_song_stub(**overrides: Any) -> YTDL:
    fields: dict = dict(
        title="Test Song",
        webpage_url="https://youtu.be/abc",
        uploader="Test Channel",
        duration_secs=242,
        position_secs=225.0,
        thumbnail="https://img/x.jpg",
        requester=SimpleNamespace(id=333, display_name="Omkar"),
    )
    fields.update(overrides)
    return cast(YTDL, SimpleNamespace(**fields))


class TestHistoryEntryFromSong:
    def test_maps_song_fields(self) -> None:
        entry = HistoryEntry.from_song(
            _history_song_stub(), guild_id=111, played_at=1752530000.0
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
        )

    def test_played_secs_is_position_reached(self) -> None:
        song = _history_song_stub(position_secs=100.4)
        assert (
            HistoryEntry.from_song(song, guild_id=111, played_at=1.0).played_secs == 100
        )

    def test_played_secs_capped_at_duration(self) -> None:
        # position can exceed duration by fractions of a frame at natural end.
        song = _history_song_stub(position_secs=243.02)
        assert (
            HistoryEntry.from_song(song, guild_id=111, played_at=1.0).played_secs == 242
        )

    def test_unknown_duration_leaves_position_uncapped(self) -> None:
        song = _history_song_stub(duration_secs=0, position_secs=99.6)
        entry = HistoryEntry.from_song(song, guild_id=111, played_at=1.0)
        assert entry.duration_secs == 0
        assert entry.played_secs == 100

    def test_no_requester_degrades_to_zero_values(self) -> None:
        song = _history_song_stub(requester=None)
        entry = HistoryEntry.from_song(song, guild_id=111, played_at=1.0)
        assert entry.requester_id == 0
        assert entry.requester_name == ""

    def test_none_metadata_degrades_to_zero_values(self) -> None:
        # yt-dlp can return None for any metadata field.
        song = _history_song_stub(
            title=None, webpage_url=None, uploader=None, thumbnail=None
        )
        entry = HistoryEntry.from_song(song, guild_id=111, played_at=1.0)
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
