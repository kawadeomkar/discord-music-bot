"""Tests for src/guild_state.py — field constants and value objects."""

import dataclasses
import logging
from types import SimpleNamespace

import pytest

from src.guild_state import GuildStateData, NowPlayingData


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
    def test_full_hash_parses_all_fields(self):
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

    def test_empty_hash_yields_zero_value_snapshot(self):
        assert GuildStateData.from_redis({}) == GuildStateData()

    def test_partial_hash_missing_fields_get_defaults(self):
        data = GuildStateData.from_redis({b"volume": b"0.8"})
        assert data.volume == 0.8
        assert data.voice_channel_id is None
        assert data.current_song_url == ""
        assert data.total_pause_seconds == 0.0

    def test_malformed_float_yields_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"play_start_epoch": b"not-a-float"})
        assert data.play_start_epoch is None
        assert "play_start_epoch" in caplog.text

    def test_malformed_int_yields_none_and_warns(self, caplog):
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"current_song_requester_id": b"abc"})
        assert data.current_song_requester_id is None
        assert "current_song_requester_id" in caplog.text

    def test_float_formatted_int_parses(self):
        data = GuildStateData.from_redis({b"voice_channel_id": b"111.0"})
        assert data.voice_channel_id == 111

    def test_snowflake_id_precision_preserved(self):
        # Discord snowflakes exceed float's 53-bit integer precision; parsing
        # via float() would corrupt 222222222222222222 to ...208.
        data = GuildStateData.from_redis(
            {b"current_song_requester_id": b"222222222222222222"}
        )
        assert data.current_song_requester_id == 222222222222222222

    def test_zero_volume_is_preserved(self):
        # Falsy-zero trap: coalescing with `or` would elevate a stored 0.0.
        data = GuildStateData.from_redis({b"volume": b"0.0"})
        assert data.volume == 0.0

    def test_missing_volume_is_none_not_default(self):
        # None means "nothing persisted" — callers skip the assignment instead
        # of clobbering live state with a fabricated 1.0 default.
        assert GuildStateData.from_redis({}).volume is None

    @pytest.mark.parametrize("raw", [b"nan", b"inf", b"-inf"])
    def test_non_finite_float_treated_as_malformed(self, raw, caplog):
        # nan/inf parse as floats but poison the position math downstream.
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"play_start_epoch": raw})
        assert data.play_start_epoch is None
        assert "play_start_epoch" in caplog.text

    def test_non_finite_int_field_treated_as_malformed(self, caplog):
        # int(float(b"inf")) raises OverflowError, not ValueError.
        with caplog.at_level(logging.WARNING, logger="src.guild_state"):
            data = GuildStateData.from_redis({b"current_song_duration": b"inf"})
        assert data.current_song_duration is None

    def test_non_utf8_bytes_degrade_instead_of_raising(self):
        # A corrupt byte in one field must not make from_redis raise — that
        # would turn get_guild_state() into None ("Redis unavailable") and
        # block recovery entirely.
        data = GuildStateData.from_redis({b"current_song_title": b"Song \xff\xfe"})
        assert data.current_song_title.startswith("Song ")

    def test_empty_uploader_coerces_to_none(self):
        data = GuildStateData.from_redis({b"current_song_uploader": b""})
        assert data.current_song_uploader is None

    def test_empty_bytes_ids_coerce_to_none(self):
        data = GuildStateData.from_redis(
            {b"current_song_duration": b"", b"current_song_requester_id": b""}
        )
        assert data.current_song_duration is None
        assert data.current_song_requester_id is None


class TestGuildStateDataProperties:
    def test_has_active_connection_true_with_both_ids(self):
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
    def test_has_active_connection_false_when_either_missing(self, kwargs):
        assert not GuildStateData(**kwargs).has_active_connection

    def test_has_crashed_song(self):
        assert GuildStateData(current_song_url="https://x").has_crashed_song
        assert not GuildStateData().has_crashed_song

    def test_was_paused_at_crash(self):
        assert GuildStateData(pause_start_epoch=1.0).was_paused_at_crash
        assert not GuildStateData().was_paused_at_crash


class TestCrashedPositionAt:
    def test_none_without_play_start_epoch(self):
        assert GuildStateData().crashed_position_at(1000.0) is None

    def test_simple_elapsed(self):
        data = GuildStateData(play_start_epoch=1000.0)
        assert data.crashed_position_at(1042.0) == 42

    def test_accumulated_pause_subtracted(self):
        data = GuildStateData(play_start_epoch=1000.0, total_pause_seconds=10.0)
        assert data.crashed_position_at(1042.0) == 32

    def test_open_pause_interval_added_on_top(self):
        data = GuildStateData(
            play_start_epoch=1000.0,
            total_pause_seconds=10.0,
            pause_start_epoch=1030.0,
        )
        # elapsed 42 − (10 accumulated + 12 still-open) = 20
        assert data.crashed_position_at(1042.0) == 20

    def test_negative_result_clamped_to_zero(self):
        # Clock skew: play_start_epoch in the future relative to `now`.
        data = GuildStateData(play_start_epoch=2000.0)
        assert data.crashed_position_at(1000.0) == 0


class TestGuildStateDataImmutability:
    def test_frozen_assignment_raises(self):
        data = GuildStateData()
        with pytest.raises(dataclasses.FrozenInstanceError):
            data.volume = 0.5  # type: ignore[misc]

    def test_slots_reject_unknown_attributes(self):
        data = GuildStateData()
        with pytest.raises((AttributeError, TypeError)):
            data.unknown_attr = 1  # type: ignore[attr-defined]


def _full_song_stub() -> SimpleNamespace:
    return SimpleNamespace(
        title="Test Song",
        webpage_url="https://youtu.be/abc",
        uploader="Test Channel",
        duration="4:00",
        thumbnail="https://img/x.jpg",
        views=1000,
        likes=50,
        abr=128,
        asr=48000,
        acodec="opus",
        requester=SimpleNamespace(id=333, mention="<@333>"),
    )


def _empty_song_stub() -> SimpleNamespace:
    return SimpleNamespace(
        title=None,
        webpage_url=None,
        uploader=None,
        duration=None,
        thumbnail=None,
        views=None,
        likes=None,
        abr=None,
        asr=None,
        acodec=None,
        requester=None,
    )


class TestNowPlayingDataFromSong:
    def test_full_song(self):
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

    def test_all_none_optionals(self):
        data = NowPlayingData.from_song(_empty_song_stub())
        assert data.title == ""
        assert data.view_count == ""
        assert data.requester_id == ""
        assert data.requester_mention == "Unknown"


class TestNowPlayingDataFromRedis:
    def test_empty_hash_returns_none(self):
        assert NowPlayingData.from_redis({}) is None

    def test_full_hash(self):
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

    def test_missing_requester_mention_defaults_to_unknown(self):
        data = NowPlayingData.from_redis({b"title": b"Test Song"})
        assert data is not None
        assert data.requester_mention == "Unknown"

    def test_round_trip_preserves_all_fields(self):
        original = NowPlayingData.from_song(_full_song_stub())
        # Encode step mirrors the wire format: to_redis_mapping() feeds HSET
        # mapping=, from_redis() consumes hgetall output (decode_responses=False).
        raw = {k.encode(): v.encode() for k, v in original.to_redis_mapping().items()}
        assert NowPlayingData.from_redis(raw) == original


class TestNowPlayingDataImmutability:
    def test_frozen_assignment_raises(self):
        data = NowPlayingData()
        with pytest.raises(dataclasses.FrozenInstanceError):
            data.title = "x"  # type: ignore[misc]
