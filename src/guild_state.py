"""
Guild state schema — single source of truth for all Redis state stored per guild.

The two Redis hashes and the queue list each have corresponding frozen dataclasses
(value objects). GuildRedisStore in redis_client.py uses these for typed reads and
field-name constants; GuildQueue in guild_queue.py converts between at-rest queue
entries and live queue items. Callers never touch raw bytes from Redis directly.

This module is pure schema: constructors, serializers, and derived read-only
properties only — no domain logic, and no runtime imports from the rest of the
project (orjson is the project-wide wire codec).
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


def _b_str(raw: dict[bytes, bytes], key: str, default: str = "") -> str:
    v = raw.get(key.encode())
    # `is None` (not truthiness): a missing key gets the default, but an
    # explicitly stored empty string b"" stays "".
    #
    # errors="replace": a corrupt (non-UTF8) byte in any one field must
    # degrade to a mangled string, not raise — a strict decode would make
    # from_redis() raise and get_guild_state() return None, misclassifying
    # corruption as "Redis unavailable" and blocking recovery until TTL expiry.
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
    # nan/inf parse fine but poison downstream arithmetic (int(nan) raises,
    # inf overflows) — treat them as malformed like any other corrupt value.
    if not math.isfinite(f):
        log.warning(f"guild_state: non-finite float for {key!r}: {v!r}")
        return None
    return f


def _b_opt_int(raw: dict[bytes, bytes], key: str) -> int | None:
    v = raw.get(key.encode())
    if v is None or v == b"":
        return None
    # Exact parse first: Discord snowflake IDs exceed float's 53-bit integer
    # precision, so routing them through float() would silently corrupt them.
    try:
        return int(v)
    except ValueError, TypeError:
        pass
    try:
        # Fallback tolerates values stored as "111.0" (already float-rounded
        # at write time, so the float round-trip loses nothing further).
        # OverflowError: int(float(b"inf")) — non-finite is malformed here too.
        return int(float(v))
    except ValueError, TypeError, OverflowError:
        log.warning(f"guild_state: malformed int for {key!r}: {v!r}")
        return None


# ── Value objects — immutable snapshots of Redis hash contents ───────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildStateData:
    """Typed snapshot of guild:{id}:state deserialized from Redis.

    All fields have Python-native types with zero-value defaults, so
    GuildStateData() is the canonical "empty hash" snapshot. Callers never
    deal with bytes or manual float()/int() coercions.

    volume is None when no value is stored (not defaulted to 1.0): the caller
    must be able to tell "nothing persisted" from "user set 1.0", so a restore
    can skip the assignment entirely instead of clobbering a concurrently
    issued -volume command with a fabricated default.
    """

    volume: float | None = None
    voice_channel_id: int | None = None
    text_channel_id: int | None = None
    current_song_url: str = ""
    current_song_title: str = ""
    current_song_duration: int | None = None
    current_song_uploader: str | None = None
    current_song_requester_id: int | None = None
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

        Named was_paused_at_crash rather than is_paused to avoid confusion with
        live playback state (vc.is_paused()). This is persisted crash-time
        state, not a reflection of the current voice client status.
        """
        return self.pause_start_epoch is not None

    def crashed_position_at(self, now: float) -> int | None:
        """Approximate playback position (seconds) at crash time, or None when
        no play_start_epoch was recorded.

        Pure function of the snapshot + a caller-supplied clock, so it is unit
        testable with zero mocks. play_start_epoch is already backdated by the
        FFmpeg -ss start offset at write time, so no offset handling is needed
        here. Callers may still cap the result at song duration
        (current_song_duration, or the cached stream duration) to prevent
        FFmpeg seeking past EOF.
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
        """Deserialize raw HGETALL output. All byte coercions are centralised
        here; an empty dict yields the zero-value snapshot."""
        # Avoid `_b_float(...) or 0.0`-style coalescing — that would elevate a
        # stored 0.0 because 0.0 is falsy. Explicit None checks instead.
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
            play_start_epoch=_b_float(raw, StateField.PLAY_START_EPOCH),
            total_pause_seconds=total_pause if total_pause is not None else 0.0,
            pause_start_epoch=_b_float(raw, StateField.PAUSE_START_EPOCH),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NowPlayingData:
    """Typed snapshot of guild:{id}:now_playing.

    Used in both directions: from_song() builds it from a live YTDL for the
    atomic start-song write, and from_redis() rebuilds it during crash
    recovery. One type, so the live embed and the recovered embed can't drift.
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
        """Canonical field extraction from a live song — the single source of
        truth for both the live embed and the Redis now_playing snapshot, so
        the two can't drift out of sync."""
        return cls(
            title=song.title or "",
            webpage_url=song.webpage_url or "",
            uploader=song.uploader or "",
            duration=song.duration or "",
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
        """Serialize to a flat string dict suitable for Redis HSET mapping.

        Spelled out rather than dataclasses.asdict(): asdict() would bind the
        wire schema to Python attribute names, so renaming an attribute would
        silently rename the Redis hash field. The explicit table pins the wire
        schema to the NowPlayingField constants.
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

    requester is stored as an ID because a live discord.Member cannot exist at
    rest; rehydration back to a QueueObject (which needs a guild for member
    resolution) happens in GuildQueue. requester_id is None only for the
    crashed-head entry derived from state via from_crashed_state() — wire
    entries always carry a real int, and snowflakes stay exact end-to-end
    (orjson native ints; never a float path).
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
        )

    @classmethod
    def from_crashed_state(
        cls, state: GuildStateData, *, position: int | None
    ) -> Self | None:
        """The crashed "current song" reborn as a queue entry — the typed
        inverse of pop_queue_and_start_song(): the current_song_* state fields
        are the fields of the entry that transaction atomically LPOPed when
        the song started. Returns None when no crashed song is recorded.

        persisted=False: the LPOP already committed, so this entry is not on
        the Redis list and the playback loop must not LPOP again for it.

        position is the caller-computed resume offset (crashed_position_at()
        plus its duration cap) — passed in so this stays a pure field mapping.
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
        )

    def to_redis(self) -> bytes:
        """Serialize to the wire format. Field table spelled out (same
        rationale as NowPlayingData.to_redis_mapping): the wire schema is
        pinned to QueueEntryField, not to Python attribute names."""
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
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchQueueEntry:
    """An unresolved search at rest ("ytsource" on the wire) — e.g. a Spotify
    playlist track awaiting yt-dlp resolution. Mirrors exactly the four
    YTSource fields the wire format persists; the remaining YTSource fields
    are reconstructed as defaults on rehydration."""

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


def parse_queue_entry(data: bytes) -> QueueEntry | None:
    """Deserialize one queue-list entry into a value object.

    Entries without a "type" field are legacy writes from before the
    discriminator existed and parse as songs. Corrupt entries (bad JSON,
    missing required fields) return None with a warning — the entry is
    dropped and the rest of the queue survives, matching the original
    _deserialize_queue_item behavior.
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
        )
    except Exception as e:
        log.warning(f"guild_state: corrupt queue entry dropped: {e}")
        return None


# ── The aggregate — a guild's persisted playback state, read as one unit ─────


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildPlaybackSnapshot:
    """A guild's persisted playback aggregate: the state hash plus the pending
    queue, read together in one round-trip (GuildRedisStore.get_playback_snapshot).

    This is the "a guild owns a queue" relationship as a type: recovery
    decisions that span both halves live here as named properties instead of
    boolean expressions at call sites.
    """

    state: GuildStateData
    queue: tuple[QueueEntry, ...] = ()

    @property
    def pending_count(self) -> int:
        return len(self.queue)

    @property
    def has_restorable_playback(self) -> bool:
        """The _restore_guild gate: True when a restart has anything to resume
        — pending queue entries or a song that was mid-play at crash time."""
        return bool(self.queue) or self.state.has_crashed_song
