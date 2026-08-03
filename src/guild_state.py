"""
Guild state schema — single source of truth for all Redis state stored per guild.

The two Redis hashes and the queue list each have corresponding frozen dataclasses
(value objects). GuildRedisStore in redis_client.py uses these for typed reads and
field-name constants; GuildQueue in guild_queue.py converts between at-rest queue
entries and live queue items. Callers never touch raw bytes from Redis directly.

This module is pure schema: constructors, serializers, derived read-only
properties, and the domain NORMALIZATION that keeps a value object inside the
column domain it is stored in (HistoryEntry.__post_init__) — but no behaviour,
and no runtime imports from the rest of the project (orjson is the project-wide
wire codec). Normalization belongs here precisely because it is schema: the type
and the domain it promises cannot be allowed to drift apart.
"""

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Self, Union

import orjson

if TYPE_CHECKING:
    from src.sources import YTSource
    from src.youtube import QueueObject, YTDL

log = logging.getLogger(__name__)


# ── guild:{id}:state hash — field name constants ─────────────────────────────


class StateField:
    VOLUME: Final[str] = "volume"
    VOICE_CHANNEL_ID: Final[str] = "voice_channel_id"
    TEXT_CHANNEL_ID: Final[str] = "text_channel_id"
    CURRENT_SONG_URL: Final[str] = "current_song_url"
    CURRENT_SONG_TITLE: Final[str] = "current_song_title"
    CURRENT_SONG_DURATION: Final[str] = "current_song_duration"
    CURRENT_SONG_UPLOADER: Final[str] = "current_song_uploader"
    CURRENT_SONG_REQUESTER_ID: Final[str] = "current_song_requester_id"
    # "1" when the playing song was queued via -playnow — preserves replace
    # semantics across a crash mid-interjection.
    CURRENT_SONG_INTERJECTED: Final[str] = "current_song_interjected"
    PLAY_START_EPOCH: Final[str] = "play_start_epoch"
    TOTAL_PAUSE_SECONDS: Final[str] = "total_pause_seconds"
    PAUSE_START_EPOCH: Final[str] = "pause_start_epoch"


# ── guild:{id}:now_playing hash — field name constants ───────────────────────


class NowPlayingField:
    TITLE: Final[str] = "title"
    WEBPAGE_URL: Final[str] = "webpage_url"
    UPLOADER: Final[str] = "uploader"
    DURATION: Final[str] = "duration"
    THUMBNAIL: Final[str] = "thumbnail"
    VIEW_COUNT: Final[str] = "view_count"
    LIKE_COUNT: Final[str] = "like_count"
    ABR: Final[str] = "abr"
    ASR: Final[str] = "asr"
    ACODEC: Final[str] = "acodec"
    REQUESTER_ID: Final[str] = "requester_id"
    REQUESTER_MENTION: Final[str] = "requester_mention"


# ── Parsing helpers (module-level; shared by both from_redis constructors) ───
#
# The bare `except A, B:` clauses below are PEP 758 (Python 3.14+)
# unparenthesized multi-exception syntax, NOT the old Python-2
# catch-one/bind-name form: they catch the tuple (A, B) and let unlisted types
# propagate. This is the concise style ruff's formatter normalizes to at
# `target-version = "py314"` (documented, intended behavior since ruff v0.15.0)
# — not a bug. requires-python is >=3.14, so it is valid on every interpreter
# this project runs on; do not re-parenthesize (ruff would strip it back).


def _b_str(raw: dict[bytes, bytes], key: str, default: str = "") -> str:
    v = raw.get(key.encode())
    # `is None`, not truthiness: a missing key gets the default, but a stored
    # empty string b"" stays "".
    #
    # errors="replace": one corrupt non-UTF8 byte must degrade to a mangled
    # string, not raise. A strict decode would make from_redis() raise and
    # get_guild_state() return None, misclassifying corruption as "Redis
    # unavailable" and blocking recovery until TTL expiry.
    return v.decode(errors="replace") if v is not None else default


def _b_float(raw: dict[bytes, bytes], key: str) -> float | None:
    v = raw.get(key.encode())
    if v is None or v == b"":
        return None
    try:
        f = float(v)
    except ValueError, TypeError:
        log.warning(f"guild_state: malformed float for {key!r}: {v!r}")
        return None
    # nan/inf parse fine but poison downstream arithmetic (int(nan) raises, inf
    # overflows) — treat them as malformed like any other corrupt value.
    if not math.isfinite(f):
        log.warning(f"guild_state: non-finite float for {key!r}: {v!r}")
        return None
    return f


def _b_opt_int(raw: dict[bytes, bytes], key: str) -> int | None:
    v = raw.get(key.encode())
    if v is None or v == b"":
        return None
    # Exact parse first: snowflake IDs exceed float's 53-bit integer precision,
    # so routing them through float() would silently corrupt them.
    try:
        return int(v)
    except ValueError, TypeError:
        pass
    try:
        # Tolerates values stored as "111.0" (already float-rounded at write
        # time, so this round-trip loses nothing further). OverflowError covers
        # int(float(b"inf")) — non-finite is malformed here too.
        return int(float(v))
    except ValueError, TypeError, OverflowError:
        log.warning(f"guild_state: malformed int for {key!r}: {v!r}")
        return None


# ── Value objects — immutable snapshots of Redis hash contents ───────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildStateData:
    """Typed snapshot of guild:{id}:state deserialized from Redis.

    Zero-value defaults throughout, so GuildStateData() is the canonical "empty
    hash" snapshot and callers never handle bytes or manual coercions.

    volume is None when nothing is stored, NOT 1.0: the caller must tell
    "nothing persisted" from "user set 1.0" so a restore can skip the assignment
    instead of clobbering a concurrent -volume with a fabricated default.
    """

    volume: float | None = None
    voice_channel_id: int | None = None
    text_channel_id: int | None = None
    current_song_url: str = ""
    current_song_title: str = ""
    current_song_duration: int | None = None
    current_song_uploader: str | None = None
    current_song_requester_id: int | None = None
    current_song_interjected: bool = False
    play_start_epoch: float | None = None
    total_pause_seconds: float = 0.0
    pause_start_epoch: float | None = None

    # Convenience properties — derived from stored fields, not stored separately.

    @property
    def has_active_connection(self) -> bool:
        """True when the bot has a persisted voice + text channel pair."""
        return self.voice_channel_id is not None and self.text_channel_id is not None

    @property
    def has_crashed_song(self) -> bool:
        """True when a song was playing when the bot last stopped."""
        return bool(self.current_song_url)

    @property
    def was_paused_at_crash(self) -> bool:
        """True when a pause_start_epoch is recorded (bot was paused at crash).

        Named for the crash, not is_paused, to keep it distinct from live state
        (vc.is_paused()): this is persisted crash-time state only.
        """
        return self.pause_start_epoch is not None

    # FIXME: Crash recovery counts bot downtime as playback position.
    # `now` is read at RESTART time while play_start_epoch dates from when the song
    # started, so 10 minutes
    # of downtime lands straight on the computed position: a song that crashed 30s in
    # comes back near its end, on the caller's duration−10s EOF cap. Only a pause already
    # active at crash time is subtracted; the crash gap is never tracked.
    # Fix: a periodic playback heartbeat in Redis, so recovery reads the last known
    # position instead of extrapolating.
    def crashed_position_at(self, now: float) -> int | None:
        """Approximate playback position (seconds) at crash time, or None when no
        play_start_epoch was recorded.

        Pure function of the snapshot plus a caller-supplied clock, so it needs no
        mocks to test. play_start_epoch is already backdated by the FFmpeg -ss
        offset at write time. Callers may still cap the result at the song duration
        (current_song_duration, or the cached stream duration) to stop FFmpeg
        seeking past EOF.
        """
        if self.play_start_epoch is None:
            return None
        elapsed = now - self.play_start_epoch
        total_pause = self.total_pause_seconds
        if self.pause_start_epoch is not None:
            total_pause += now - self.pause_start_epoch
        return max(0, int(elapsed - total_pause))

    @classmethod
    def from_redis(cls, raw: dict[bytes, bytes]) -> Self:
        """Deserialize raw HGETALL output. All byte coercions are centralised here;
        an empty dict yields the zero-value snapshot."""
        # No `_b_float(...) or 0.0` coalescing — 0.0 is falsy, so that would
        # elevate a stored 0.0 to the default. Explicit None checks instead.
        total_pause = _b_float(raw, StateField.TOTAL_PAUSE_SECONDS)
        return cls(
            volume=_b_float(raw, StateField.VOLUME),
            voice_channel_id=_b_opt_int(raw, StateField.VOICE_CHANNEL_ID),
            text_channel_id=_b_opt_int(raw, StateField.TEXT_CHANNEL_ID),
            current_song_url=_b_str(raw, StateField.CURRENT_SONG_URL),
            current_song_title=_b_str(raw, StateField.CURRENT_SONG_TITLE),
            current_song_duration=_b_opt_int(raw, StateField.CURRENT_SONG_DURATION),
            current_song_uploader=_b_str(raw, StateField.CURRENT_SONG_UPLOADER) or None,
            current_song_requester_id=_b_opt_int(
                raw, StateField.CURRENT_SONG_REQUESTER_ID
            ),
            current_song_interjected=(
                _b_str(raw, StateField.CURRENT_SONG_INTERJECTED) == "1"
            ),
            play_start_epoch=_b_float(raw, StateField.PLAY_START_EPOCH),
            total_pause_seconds=total_pause if total_pause is not None else 0.0,
            pause_start_epoch=_b_float(raw, StateField.PAUSE_START_EPOCH),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NowPlayingData:
    """Typed snapshot of guild:{id}:now_playing.

    Bidirectional: from_song() builds it from a live YTDL for the atomic
    start-song write, from_redis() rebuilds it during crash recovery. One type,
    so the live and recovered embeds can't drift.
    """

    title: str = ""
    webpage_url: str = ""
    uploader: str = ""
    duration: str = ""
    thumbnail: str = ""
    view_count: str = ""
    like_count: str = ""
    abr: str = ""
    asr: str = ""
    acodec: str = ""
    requester_id: str = ""
    requester_mention: str = "Unknown"  # matches the write path's default

    @classmethod
    def from_song(cls, song: YTDL) -> Self:
        """Canonical field extraction from a live song — one source of truth for
        the live embed and the Redis snapshot, so the two can't drift."""
        return cls(
            title=song.title or "",
            webpage_url=song.webpage_url or "",
            uploader=song.uploader or "",
            # Empty for unknown duration (livestream, missing metadata) rather
            # than the "0:00" fmt_duration renders — mirrors the live embed, which
            # draws no bar when duration_secs <= 0. The recovered embed keys its
            # Duration line off this being truthy.
            duration=song.duration if song.duration_secs > 0 else "",
            thumbnail=song.thumbnail or "",
            view_count=str(song.views) if song.views is not None else "",
            like_count=str(song.likes) if song.likes is not None else "",
            abr=str(song.abr) if song.abr is not None else "",
            asr=str(song.asr) if song.asr is not None else "",
            acodec=song.acodec or "",
            requester_id=str(song.requester.id) if song.requester else "",
            requester_mention=song.requester.mention if song.requester else "Unknown",
        )

    @classmethod
    def from_redis(cls, raw: dict[bytes, bytes]) -> Self | None:
        """Deserialize raw HGETALL output. Returns None if the hash is empty
        (the hash is DELETE'd wholesale on song end, so empty == no song)."""
        if not raw:
            return None
        return cls(
            title=_b_str(raw, NowPlayingField.TITLE),
            webpage_url=_b_str(raw, NowPlayingField.WEBPAGE_URL),
            uploader=_b_str(raw, NowPlayingField.UPLOADER),
            duration=_b_str(raw, NowPlayingField.DURATION),
            thumbnail=_b_str(raw, NowPlayingField.THUMBNAIL),
            view_count=_b_str(raw, NowPlayingField.VIEW_COUNT),
            like_count=_b_str(raw, NowPlayingField.LIKE_COUNT),
            abr=_b_str(raw, NowPlayingField.ABR),
            asr=_b_str(raw, NowPlayingField.ASR),
            acodec=_b_str(raw, NowPlayingField.ACODEC),
            requester_id=_b_str(raw, NowPlayingField.REQUESTER_ID),
            requester_mention=_b_str(
                raw, NowPlayingField.REQUESTER_MENTION, default="Unknown"
            ),
        )

    def to_redis_mapping(self) -> dict[str, str]:
        """Serialize to a flat string dict for Redis HSET mapping.

        Spelled out rather than dataclasses.asdict(), which would bind the wire
        schema to Python attribute names — renaming an attribute would silently
        rename the hash field. This table pins it to the NowPlayingField constants.
        """
        return {
            NowPlayingField.TITLE: self.title,
            NowPlayingField.WEBPAGE_URL: self.webpage_url,
            NowPlayingField.UPLOADER: self.uploader,
            NowPlayingField.DURATION: self.duration,
            NowPlayingField.THUMBNAIL: self.thumbnail,
            NowPlayingField.VIEW_COUNT: self.view_count,
            NowPlayingField.LIKE_COUNT: self.like_count,
            NowPlayingField.ABR: self.abr,
            NowPlayingField.ASR: self.asr,
            NowPlayingField.ACODEC: self.acodec,
            NowPlayingField.REQUESTER_ID: self.requester_id,
            NowPlayingField.REQUESTER_MENTION: self.requester_mention,
        }


# ── guild:{id}:queue list — JSON field name constants ────────────────────────


class QueueEntryField:
    TYPE: Final[str] = "type"
    # "qobj" entries
    WEBPAGE_URL: Final[str] = "webpage_url"
    TITLE: Final[str] = "title"
    REQUESTER_ID: Final[str] = "requester_id"
    TS: Final[str] = "ts"
    USER_INPUT: Final[str] = "user_input"
    DURATION: Final[str] = "duration"
    UPLOADER: Final[str] = "uploader"
    THUMBNAIL: Final[str] = "thumbnail"
    PERSISTED: Final[str] = "persisted"
    # -playnow flags — absent on pre-feature entries, parsed as False.
    INTERJECTED: Final[str] = "interjected"
    IS_RESUME: Final[str] = "is_resume"
    START_PAUSED: Final[str] = "start_paused"
    # "ytsource" entries
    YTSEARCH: Final[str] = "ytsearch"
    URL: Final[str] = "url"
    PROCESS: Final[str] = "process"


# Wire discriminator values — kept verbatim from the original serializer so
# entries written before and after the migration stay mutually readable.
_ENTRY_TYPE_SONG: Final[str] = "qobj"
_ENTRY_TYPE_SEARCH: Final[str] = "ytsource"


# ── Queue-entry value objects — the guild:{id}:queue list at rest ────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class SongQueueEntry:
    """A resolved song at rest ("qobj" on the wire) — the pure-data twin of
    src.youtube.QueueObject.

    requester is an ID because a live discord.Member cannot exist at rest;
    rehydration (which needs a guild to resolve members) happens in GuildQueue.
    requester_id is None only for the crashed-head entry from
    from_crashed_state() — wire entries always carry a real int, and snowflakes
    stay exact end-to-end (orjson native ints, never a float path).
    """

    webpage_url: str
    title: str
    requester_id: int | None
    ts: int | None = None
    user_input: str | None = None
    duration: int | None = None
    uploader: str | None = None
    thumbnail: str | None = None
    persisted: bool = True
    # -playnow flags — see the matching QueueObject field comments.
    interjected: bool = False
    is_resume: bool = False
    start_paused: bool = False

    @classmethod
    def from_queue_object(cls, item: QueueObject) -> Self:
        """Snapshot a live queue item for persistence."""
        return cls(
            webpage_url=item.webpage_url,
            title=item.title,
            requester_id=item.requester.id,
            ts=item.ts,
            user_input=item.user_input,
            duration=item.duration,
            uploader=item.uploader,
            thumbnail=item.thumbnail,
            persisted=item.persisted,
            interjected=item.interjected,
            is_resume=item.is_resume,
            start_paused=item.start_paused,
        )

    @classmethod
    def from_song(cls, song: YTDL) -> Self:
        """The queue-entry view of a now-playing song — write-side twin of
        from_crashed_state(). The loop hands this to the atomic start-song
        transaction, which parks the fields in the state hash as current_song_*:

            from_song → HSET state → crash → from_crashed_state → re-queue
        """
        return cls(
            webpage_url=song.webpage_url or "",
            title=song.title or "",
            requester_id=song.requester.id if song.requester else None,
            duration=song.duration_secs or None,
            uploader=song.uploader,
            interjected=song.interjected,
        )

    @classmethod
    def from_crashed_state(
        cls, state: GuildStateData, *, position: int | None
    ) -> Self | None:
        """The crashed "current song" reborn as a queue entry — the typed inverse
        of pop_queue_and_start_song(), whose current_song_* fields ARE the entry
        it LPOPed. None when no crashed song is recorded.

        persisted=False: that LPOP already committed, so this entry is not on the
        Redis list and the loop must not LPOP again for it.

        `position` is the caller-computed resume offset (crashed_position_at()
        plus its duration cap), passed in so this stays a pure field mapping.
        """
        if not state.has_crashed_song:
            return None
        return cls(
            webpage_url=state.current_song_url,
            title=state.current_song_title,
            requester_id=state.current_song_requester_id,
            ts=position,
            duration=state.current_song_duration,
            uploader=state.current_song_uploader,
            persisted=False,
            # A crash mid-interjection must not demote the recovered song to a
            # normal one: a -playnow after recovery still replaces it (no resume
            # entry) instead of stacking one.
            interjected=state.current_song_interjected,
        )

    def to_redis(self) -> bytes:
        """Serialize to the wire format. Field table spelled out for the same
        reason as NowPlayingData.to_redis_mapping: the wire schema is pinned to
        QueueEntryField, not to Python attribute names."""
        return orjson.dumps(
            {
                QueueEntryField.TYPE: _ENTRY_TYPE_SONG,
                QueueEntryField.WEBPAGE_URL: self.webpage_url,
                QueueEntryField.TITLE: self.title,
                QueueEntryField.REQUESTER_ID: self.requester_id,
                QueueEntryField.TS: self.ts,
                QueueEntryField.USER_INPUT: self.user_input,
                QueueEntryField.DURATION: self.duration,
                QueueEntryField.UPLOADER: self.uploader,
                QueueEntryField.THUMBNAIL: self.thumbnail,
                QueueEntryField.PERSISTED: self.persisted,
                QueueEntryField.INTERJECTED: self.interjected,
                QueueEntryField.IS_RESUME: self.is_resume,
                QueueEntryField.START_PAUSED: self.start_paused,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchQueueEntry:
    """An unresolved search at rest ("ytsource" on the wire) — e.g. a Spotify
    playlist track awaiting yt-dlp resolution. Holds exactly the four YTSource
    fields the wire persists; the rest default on rehydration."""

    ytsearch: str | None = None
    url: str | None = None
    process: bool | None = None
    ts: int | None = None

    @classmethod
    def from_ytsource(cls, source: YTSource) -> Self:
        return cls(
            ytsearch=source.ytsearch,
            url=source.url,
            process=source.process,
            ts=source.ts,
        )

    def to_redis(self) -> bytes:
        return orjson.dumps(
            {
                QueueEntryField.TYPE: _ENTRY_TYPE_SEARCH,
                QueueEntryField.YTSEARCH: self.ytsearch,
                QueueEntryField.URL: self.url,
                QueueEntryField.PROCESS: self.process,
                QueueEntryField.TS: self.ts,
            }
        )


QueueEntry = Union[SongQueueEntry, SearchQueueEntry]


# `bytes | str` matches orjson.loads() and redis-py's declared LRANGE return.
# Narrowing to bytes forced every caller to cast for a call always fine at runtime
# (parse_history_entry is the sibling).
def parse_queue_entry(data: bytes | str) -> QueueEntry | None:
    """Deserialize one queue-list entry into a value object.

    The "type" field discriminates searches from songs. Corrupt entries (bad
    JSON, missing required fields) return None with a warning, so one bad entry
    is dropped and the rest of the queue survives.
    """
    try:
        d = orjson.loads(data)
        if d.get(QueueEntryField.TYPE) == _ENTRY_TYPE_SEARCH:
            return SearchQueueEntry(
                ytsearch=d.get(QueueEntryField.YTSEARCH),
                url=d.get(QueueEntryField.URL),
                process=d.get(QueueEntryField.PROCESS),
                ts=d.get(QueueEntryField.TS),
            )
        return SongQueueEntry(
            webpage_url=d[QueueEntryField.WEBPAGE_URL],
            title=d[QueueEntryField.TITLE],
            requester_id=d[QueueEntryField.REQUESTER_ID],
            ts=d.get(QueueEntryField.TS),
            user_input=d.get(QueueEntryField.USER_INPUT),
            duration=d.get(QueueEntryField.DURATION),
            uploader=d.get(QueueEntryField.UPLOADER),
            thumbnail=d.get(QueueEntryField.THUMBNAIL),
            persisted=d.get(QueueEntryField.PERSISTED, True),
            interjected=d.get(QueueEntryField.INTERJECTED, False),
            is_resume=d.get(QueueEntryField.IS_RESUME, False),
            start_paused=d.get(QueueEntryField.START_PAUSED, False),
        )
    except Exception as e:
        log.warning(f"guild_state: corrupt queue entry dropped: {e}")
        return None


# ── guild:{id}:history list — wire format ────────────────────────────────────
# One JSON object of HistoryEntryField keys per entry, newest-first.


class HistoryEntryField:
    GUILD_ID: Final[str] = "guild_id"
    TITLE: Final[str] = "title"
    WEBPAGE_URL: Final[str] = "webpage_url"
    DURATION_SECS: Final[str] = "duration_secs"
    PLAYED_SECS: Final[str] = "played_secs"
    REQUESTER_ID: Final[str] = "requester_id"
    REQUESTER_NAME: Final[str] = "requester_name"
    THUMBNAIL: Final[str] = "thumbnail"
    UPLOADER: Final[str] = "uploader"
    PLAYED_AT: Final[str] = "played_at"
    MESSAGE_ID: Final[str] = "message_id"


# The play_history column domain (migrations/0001_play_history.sql), mirrored
# here because HistoryEntry is now the thing that guarantees it. Kept next to the
# dataclass rather than in history_archive.py so the type and its domain cannot
# drift apart, and so this module stays free of runtime imports from the rest of
# the project.
_TEXT_FIELDS: Final[tuple[str, ...]] = (
    "title",
    "webpage_url",
    "requester_name",
    "thumbnail",
    "uploader",
)
_INT4_FIELDS: Final[tuple[str, ...]] = ("duration_secs", "played_secs")
_INT8_FIELDS: Final[tuple[str, ...]] = ("guild_id", "requester_id", "message_id")
_INT4_MAX: Final[int] = 2**31 - 1
_INT8_MAX: Final[int] = 2**63 - 1
_TS_MAX: Final[float] = 253402300799.0  # 9999-12-31T23:59:59Z, the timestamptz ceiling


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryEntry:
    """One played song at rest — an element of guild:{id}:history.

    Zero-values mean "unknown": fields absent on the wire default on parse, and
    the display layer degrades accordingly. The field set deliberately matches
    the Postgres play_history row.

    guild_id is redundant on the per-guild display list (the key carries it) but
    load-bearing on the global history:outbox stream, where entries from all
    guilds interleave and the drainer maps each to a Postgres row. Entries
    written before the field existed parse as guild_id=0.

    message_id is a WEAK reference by design — the NP host migrates across
    messages during one song and a dedicated host is deleted when retired, so
    this records which message hosted the block when the song ended, not a
    message guaranteed to still exist. That is why it is neither a foreign key
    nor part of play_history_dedup.
    """

    guild_id: int = 0
    title: str = ""
    webpage_url: str = ""  # YouTube link used
    duration_secs: int = 0  # full song length; 0 = unknown
    played_secs: int = 0  # audio position reached when the song ended
    requester_id: int = 0  # 0 = unknown
    requester_name: str = ""  # display_name at play time; survives member departure
    thumbnail: str = ""
    uploader: str = ""
    played_at: float = 0.0  # unix epoch at song end; drives <t:…:f>
    message_id: int = 0  # NP host at song end; 0 = unknown (see class docstring)

    def __post_init__(self) -> None:
        """Normalize into the play_history column domain at construction.

        This is the schema lock: an instance of this class is, by construction, a
        row Postgres accepts. Every producer routes through here — from_song,
        parse_history_entry, _row_to_entry, dataclasses.replace, the backfill — so
        no consumer re-proves it and the archive holds no opinion about data.

        ONE FIELD IS DELIBERATELY EXEMPT: the clamp floors every integer at 0, but
        play_history's CHECK on guild_id is strictly `> 0`, so a HistoryEntry
        holding guild_id 0 is constructible and NOT insertable. Clamping up to 1
        would file an unattributable play into a real guild's history, where it is
        indistinguishable from a genuine play; relaxing the CHECK to `>= 0` would
        make guild 0 a permanent bucket of orphans no read path excludes. Refusing
        at the database sends exactly those entries to play_history_rejected, where
        a row means "a producer stopped stamping guild_id" — the actual defect. No
        real producer can reach it today (from_song takes guild_id keyword-only and
        Discord snowflakes are positive).

        The four vectors closed here are the ONLY ways a wire-parseable entry could
        fail an INSERT. Everything else the column types could refuse — lone
        surrogates, non-finite floats, integers past 64 bits — orjson already
        refuses to encode.

        Written as range tests rather than pairs of bound checks because a chained
        comparison is False for NaN, which lands NaN on the sentinel. That arm is
        unreachable from the wire (orjson emits null, the parser reads 0.0) and
        reachable from from_song, which is why it belongs on the type.

        TOTAL BY DESIGN — this never raises. It also runs on the READ path
        (_row_to_entry, and the backfill's dataclasses.replace), and stored rows
        predate it, so a validator that rejected them would break -history for
        precisely the guilds with the most history. Strictness lives in the DB
        CHECK constraints instead, where a violation means "this validator
        regressed" and cannot retroactively invalidate durable data.

        No type coercion: HistoryEntry(title=None) raising TypeError out of the
        `in` test below is the correct outcome — coercing would turn a caught bug
        into a stored row reading "None".
        """
        for name in _TEXT_FIELDS:
            value: str = getattr(self, name)
            if "\x00" in value:
                object.__setattr__(self, name, value.replace("\x00", ""))
        for name, ceiling in (
            *((f, _INT4_MAX) for f in _INT4_FIELDS),
            *((f, _INT8_MAX) for f in _INT8_FIELDS),
        ):
            value_int: int = getattr(self, name)
            if not 0 <= value_int <= ceiling:
                object.__setattr__(self, name, min(max(value_int, 0), ceiling))
        if not 0.0 <= self.played_at <= _TS_MAX:
            object.__setattr__(self, "played_at", 0.0)

    @classmethod
    def from_song(
        cls, song: YTDL, *, guild_id: int, played_at: float, message_id: int
    ) -> Self:
        """Canonical extraction from a finished song. played_at is a
        caller-supplied clock (as in crashed_position_at) so this stays a pure
        field mapping. guild_id is required (keyword) because the song object
        doesn't carry it and a forgotten stamp would silently write
        unattributable outbox entries.

        message_id is required for the same reason, and more strictly: nothing
        downstream can catch a forgotten one. play_history's CHECK is
        `message_id >= 0`, so a missed stamp inserts cleanly as 0 and is
        permanently indistinguishable from a song that genuinely had no host.
        Pass 0 explicitly for that case.

        played_secs is the position reached (start_offset + audio delivered),
        capped at duration when known. For a -playnow-interrupted song recorded
        once at its resume tail's end, the tail's position spans the full
        listened range — the desirable answer. A ?t= song includes the skip;
        accepted.
        """
        played = round(song.position_secs)
        duration = song.duration_secs or 0
        if duration:
            played = min(played, duration)
        return cls(
            guild_id=guild_id,
            title=song.title or "",
            webpage_url=song.webpage_url or "",
            duration_secs=duration,
            played_secs=played,
            requester_id=song.requester.id if song.requester else 0,
            requester_name=song.requester.display_name if song.requester else "",
            thumbnail=song.thumbnail or "",
            uploader=song.uploader or "",
            played_at=played_at,
            message_id=message_id,
        )

    def to_redis(self) -> bytes:
        """Serialize to the wire format. Field table spelled out for the same
        reason as NowPlayingData.to_redis_mapping: the wire schema is pinned to
        HistoryEntryField, not to Python attribute names."""
        return orjson.dumps(
            {
                HistoryEntryField.GUILD_ID: self.guild_id,
                HistoryEntryField.TITLE: self.title,
                HistoryEntryField.WEBPAGE_URL: self.webpage_url,
                HistoryEntryField.DURATION_SECS: self.duration_secs,
                HistoryEntryField.PLAYED_SECS: self.played_secs,
                HistoryEntryField.REQUESTER_ID: self.requester_id,
                HistoryEntryField.REQUESTER_NAME: self.requester_name,
                HistoryEntryField.THUMBNAIL: self.thumbnail,
                HistoryEntryField.UPLOADER: self.uploader,
                HistoryEntryField.PLAYED_AT: self.played_at,
                HistoryEntryField.MESSAGE_ID: self.message_id,
            }
        )


def serialize_history_entry(entry: HistoryEntry) -> bytes:
    return entry.to_redis()


def parse_history_entry(data: bytes | str) -> HistoryEntry | None:
    """Deserialize one history-list entry. Corrupt entries (bad JSON, wrong type,
    malformed fields) return None with a warning and are dropped, as in
    parse_queue_entry. Unknown keys are ignored and missing keys default, so
    mixed-build readers stay tolerant in both directions."""
    try:
        entry = orjson.loads(data)
    except Exception as e:
        log.warning(f"guild_state: corrupt history entry dropped: {e}")
        return None
    if not isinstance(entry, dict):
        log.warning(
            f"guild_state: corrupt history entry dropped: unexpected JSON type ({type(entry).__name__})"
        )
        return None
    try:
        return HistoryEntry(
            guild_id=int(entry.get(HistoryEntryField.GUILD_ID) or 0),
            title=str(entry.get(HistoryEntryField.TITLE) or ""),
            webpage_url=str(entry.get(HistoryEntryField.WEBPAGE_URL) or ""),
            duration_secs=int(entry.get(HistoryEntryField.DURATION_SECS) or 0),
            played_secs=int(entry.get(HistoryEntryField.PLAYED_SECS) or 0),
            requester_id=int(entry.get(HistoryEntryField.REQUESTER_ID) or 0),
            requester_name=str(entry.get(HistoryEntryField.REQUESTER_NAME) or ""),
            thumbnail=str(entry.get(HistoryEntryField.THUMBNAIL) or ""),
            uploader=str(entry.get(HistoryEntryField.UPLOADER) or ""),
            played_at=float(entry.get(HistoryEntryField.PLAYED_AT) or 0.0),
            message_id=int(entry.get(HistoryEntryField.MESSAGE_ID) or 0),
        )
    except Exception as e:
        log.warning(f"guild_state: corrupt history entry dropped: {e}")
        return None


# ── The aggregate — a guild's persisted playback state, read as one unit ─────


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildPlaybackSnapshot:
    """A guild's complete persisted playback aggregate — state hash, pending
    queue, now-playing snapshot, played-song history — read together in one
    pipelined round-trip (GuildRedisStore.get_playback_snapshot).

    The "a guild owns a queue (and a history)" relationship as a type: recovery
    decisions spanning the halves live here as named properties rather than as
    boolean expressions at call sites.
    """

    state: GuildStateData
    queue: tuple[QueueEntry, ...] = ()
    # None when no song was playing (the hash is DELETE'd wholesale on song
    # end, so empty == no song — same contract as NowPlayingData.from_redis).
    now_playing: NowPlayingData | None = None
    # Newest-first, as stored (GuildHistory.restore() handles the reversal).
    history: tuple[HistoryEntry, ...] = ()

    @property
    def pending_count(self) -> int:
        return len(self.queue)

    @property
    def has_restorable_playback(self) -> bool:
        """The _restore_guild gate: True when a restart has anything to resume
        — pending queue entries or a song that was mid-play at crash time."""
        return bool(self.queue) or self.state.has_crashed_song


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildRecoveryGate:
    """The minimal read `_restore_guild` needs to decide whether to reconnect:
    the state hash plus the pending queue's *length* — never its contents.

    _restore_guild consults only the connection gate, the channel IDs, and
    whether there is anything to resume; _restore_state re-reads the full
    GuildPlaybackSnapshot after a successful voice connect. Reading only the
    length keeps a -stopped guild's (possibly long) leftover queue off the wire
    on every on_ready (see GuildRedisStore.get_recovery_gate).
    """

    state: GuildStateData
    pending_count: int = 0

    @property
    def has_restorable_playback(self) -> bool:
        """True when a restart has anything to resume — pending entries or a song
        mid-play at crash time. Mirrors
        GuildPlaybackSnapshot.has_restorable_playback, over the queue length."""
        return self.pending_count > 0 or self.state.has_crashed_song
