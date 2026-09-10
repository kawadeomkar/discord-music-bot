"""
Guild state schema — single source of truth for all Redis state stored per guild.

Each Redis hash and list has a frozen value object here; GuildRedisStore reads
and writes through them and GuildQueue converts between at-rest entries and live
items, so no caller touches raw bytes. Pure schema: constructors, serializers,
derived properties, and the domain normalization that keeps a value object inside
the column it is stored in (HistoryEntry.__post_init__). No runtime imports from
the rest of the project.
"""

import logging
import math
import re
from zoneinfo import ZoneInfo, available_timezones
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Final, Self, Union

import orjson

if TYPE_CHECKING:
    from src.sources import YTSource
    from src.youtube import QueueObject, YTDL

log = logging.getLogger(__name__)


# ── Pure-analytics values, grouped ───────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class Analytics:
    """Values carried on live queue objects (QueueObject, YTSource, YTDL) for
    storage alone — read only to serialize or to carry onto the next object; a
    field anything branches on or renders belongs elsewhere. In-memory shape
    only: wire entries and play_history columns stay FLAT. Frozen, because carry
    sites alias one instance across a resume tail and its source."""

    # Unix epoch when the user ASKED: the command message's snowflake time
    # (Discord's clock, so played_at - queued_at can go slightly negative).
    # 0.0 = unknown (pre-feature wire entries).
    queued_at: float
    # Songs ahead at ask time, counting the one playing (0 = played immediately).
    # Read once at dispatch, so it is approximate against the insert.
    queue_position: int


# What a pre-feature wire entry rehydrates as, and the default on live objects
# whose construction site cannot know the values yet.
ANALYTICS_ZERO: Final[Analytics] = Analytics(queued_at=0.0, queue_position=0)


# ── guild:{id}:state hash — field name constants ─────────────────────────────


class StateField:
    # LEGACY: volume lives in guild:{id}:config (GuildConfig), read first. Still
    # read AND written for one release so a rollback finds a fresh value. Drop
    # this, GuildStateData.volume and set_volume's second write together.
    VOLUME: Final[str] = "volume"
    VOICE_CHANNEL_ID: Final[str] = "voice_channel_id"
    TEXT_CHANNEL_ID: Final[str] = "text_channel_id"
    CURRENT_SONG_URL: Final[str] = "current_song_url"
    CURRENT_SONG_TITLE: Final[str] = "current_song_title"
    CURRENT_SONG_DURATION: Final[str] = "current_song_duration"
    CURRENT_SONG_UPLOADER: Final[str] = "current_song_uploader"
    CURRENT_SONG_REQUESTER_ID: Final[str] = "current_song_requester_id"
    # "1" when the playing song was queued by an interjection (attribution only).
    CURRENT_SONG_INTERJECTED: Final[str] = "current_song_interjected"
    # "1" when the playing song is an interjection's resume tail / was parked
    # paused. is_resume drives the announcement, _remaining_secs and NP-card
    # cleanup.
    CURRENT_SONG_IS_RESUME: Final[str] = "current_song_is_resume"
    CURRENT_SONG_START_PAUSED: Final[str] = "current_song_start_paused"
    # Set once at ask time and carried, never rewritten, so a crash-recovered
    # song still archives what it was originally queued with.
    CURRENT_SONG_QUEUED_AT: Final[str] = "current_song_queued_at"
    CURRENT_SONG_QUEUE_POSITION: Final[str] = "current_song_queue_position"
    CURRENT_SONG_QUERY_SOURCE: Final[str] = "current_song_query_source"
    # What the user typed, so -remove <collection link> can take out a
    # crash-recovered head.
    CURRENT_SONG_USER_INPUT: Final[str] = "current_song_user_input"
    # When the audio started. Not PLAY_START_EPOCH (backdated by -ss) and not
    # derivable from this run's clock: a resume tail inherits an earlier
    # fragment's value.
    CURRENT_SONG_PLAYED_AT: Final[str] = "current_song_played_at"
    # LEGACY, all three: read only by _legacy_wall_clock_position_at, still
    # written so a rollback recovers. Drop with that method and on_pause/
    # on_resume one release after the heartbeat ships.
    PLAY_START_EPOCH: Final[str] = "play_start_epoch"
    TOTAL_PAUSE_SECONDS: Final[str] = "total_pause_seconds"
    PAUSE_START_EPOCH: Final[str] = "pause_start_epoch"
    # The recorded playback position, read with no wall-clock arithmetic.
    LAST_POSITION_SECS: Final[str] = "last_position_secs"
    # When that position was recorded. Never an addend — it lets
    # _heartbeat_predates_song refuse a position belonging to an earlier song.
    LAST_HEARTBEAT_EPOCH: Final[str] = "last_heartbeat_epoch"


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


# ── Parsing helpers (shared by the from_redis constructors) ──────────────────
#
# `except A, B:` below is PEP 758 (Python 3.14+) tuple-catch syntax, normalized
# by ruff at target-version py314; do not re-parenthesize.


def _b_str(raw: dict[bytes, bytes], key: str, default: str = "") -> str:
    v = raw.get(key.encode())
    # `is None`, not truthiness: a stored b"" stays "". errors="replace" so one
    # corrupt byte degrades to a mangled string rather than raising out of
    # from_redis(), which would read as "Redis unavailable" and block recovery.
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
    # nan/inf parse fine but poison downstream arithmetic.
    if not math.isfinite(f):
        log.warning(f"guild_state: non-finite float for {key!r}: {v!r}")
        return None
    return f


def _b_opt_int(raw: dict[bytes, bytes], key: str) -> int | None:
    v = raw.get(key.encode())
    if v is None or v == b"":
        return None
    # Exact parse first: snowflake IDs exceed float's 53-bit integer precision.
    try:
        return int(v)
    except ValueError, TypeError:
        pass
    try:
        # Tolerates "111.0" (float-rounded at write time). OverflowError covers
        # int(float(b"inf")).
        return int(float(v))
    except ValueError, TypeError, OverflowError:
        log.warning(f"guild_state: malformed int for {key!r}: {v!r}")
        return None


# ── Value objects — immutable snapshots of Redis hash contents ───────────────


class ConfigField:
    """Wire field names for guild:{id}:config, spelled out so renaming a Python
    attribute can never silently rename a Redis field."""

    DEBUG_MODE: Final[str] = "debug_mode"
    VOLUME: Final[str] = "volume"
    TIMEZONE: Final[str] = "timezone"


# The zone every guild renders ETAs in until it picks one; the schema layer
# validates against the same default it hands back.
DEFAULT_TIMEZONE: Final[str] = "America/Los_Angeles"

# Zone names already proven unusable on this host. ZoneInfo caches successful
# lookups only; a failed one is a filesystem miss plus a log line on every
# render. Capped because this key outlives builds and can be hand-edited.
_UNUSABLE_ZONES: Final[set[str]] = set()
_MAX_UNUSABLE_ZONES: Final[int] = 256

# The longest real IANA name is 32 chars ("America/Argentina/ComodRivadavia").
_MAX_TIMEZONE_NAME: Final[int] = 64


@lru_cache(maxsize=1)
def _known_zones() -> frozenset[str]:
    """Every zone this host can name. available_timezones() walks the whole tz
    database, and the answer cannot change without a restart."""
    return frozenset(available_timezones())


def valid_timezone(name: str) -> bool:
    """The WRITE boundary's check: a bad name stored unvalidated fails silently
    (the write succeeds, the command reports success, ETAs stay on the default
    forever). Membership, not `ZoneInfo(name)`: ZoneInfo resolves BY PATH, so
    `zone.tab` and `leapseconds` construct fine without being zones."""
    return 0 < len(name) <= _MAX_TIMEZONE_NAME and name in _known_zones()


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildConfig:
    """A guild's DURABLE preferences, in its own key: guild:{id}:state carries a
    24h TTL, and a setting stored there reverts on any guild idle for a day.
    Every field is Optional because absent means "follow the host default",
    which is not the same as an explicitly chosen value — a guild that turned
    debug off while the host default is on must stay off, and restore must skip
    an unset volume rather than clobber a concurrent -volume with 1.0."""

    debug_mode: bool | None = None
    volume: float | None = None
    # An IANA name, not a ZoneInfo: the wire stays human-readable. Resolved by
    # tzinfo(), which is also where an unusable name degrades.
    timezone: str | None = None

    def to_redis(self) -> dict[str, str]:
        """Only fields with a value: an unset field is ABSENT from the hash, so
        "never chose" survives a round trip."""
        mapping: dict[str, str] = {}
        if self.debug_mode is not None:
            mapping[ConfigField.DEBUG_MODE] = "1" if self.debug_mode else "0"
        if self.volume is not None:
            mapping[ConfigField.VOLUME] = str(self.volume)
        if self.timezone is not None:
            mapping[ConfigField.TIMEZONE] = self.timezone
        return mapping

    def tzinfo(self) -> ZoneInfo:
        """The guild's zone, or the default when it has not chosen a usable one.
        Resolved at read time because the tz database is a property of the host:
        a name that resolved when set can stop resolving after a base-image
        change, and a default beats a render path that raises."""
        # tzdata is a declared dependency, so the default always resolves.
        if self.timezone is None or self.timezone in _UNUSABLE_ZONES:
            return ZoneInfo(DEFAULT_TIMEZONE)
        try:
            return ZoneInfo(self.timezone)
        except Exception:  # noqa: BLE001 — any unusable name falls back
            log.warning(
                f"guild_state: unknown timezone {self.timezone!r}; using default"
            )
            if len(_UNUSABLE_ZONES) < _MAX_UNUSABLE_ZONES:
                _UNUSABLE_ZONES.add(self.timezone)
            return ZoneInfo(DEFAULT_TIMEZONE)

    @classmethod
    def from_redis(cls, raw: dict[bytes, bytes]) -> Self:
        """Deserialize raw HGETALL output; an empty dict yields all-unset. An
        unparseable field reads as unset rather than raising: this key outlives
        builds, and one bad field must not cost the guild its whole config."""
        return cls(
            debug_mode={"1": True, "0": False}.get(_b_str(raw, ConfigField.DEBUG_MODE)),
            volume=_b_float(raw, ConfigField.VOLUME),
            timezone=_b_str(raw, ConfigField.TIMEZONE) or None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildStateData:
    """Typed snapshot of guild:{id}:state. Zero-value defaults throughout, so
    GuildStateData() is the "empty hash" snapshot. volume is None when nothing
    is stored, not 1.0, so a restore can skip the assignment instead of
    clobbering a concurrent -volume."""

    volume: float | None = None
    voice_channel_id: int | None = None
    text_channel_id: int | None = None
    current_song_url: str = ""
    current_song_title: str = ""
    current_song_duration: int | None = None
    current_song_uploader: str | None = None
    current_song_requester_id: int | None = None
    current_song_interjected: bool = False
    current_song_is_resume: bool = False
    current_song_start_paused: bool = False
    current_song_queued_at: float = 0.0
    current_song_queue_position: int = 0
    current_song_query_source: str = ""
    # None, not "": absent means a pre-migration entry, and an empty needle must
    # never be what -remove matches on. parse_queue_entry draws the same line.
    current_song_user_input: str | None = None
    current_song_played_at: float = 0.0
    play_start_epoch: float | None = None
    total_pause_seconds: float = 0.0
    pause_start_epoch: float | None = None
    last_position_secs: float | None = None
    last_heartbeat_epoch: float | None = None

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
        """True when a pause_start_epoch is recorded — persisted crash-time
        state, distinct from the live vc.is_paused()."""
        return self.pause_start_epoch is not None

    def crashed_position_at(self, now: float) -> int | None:
        """Playback position (seconds) at the last recorded heartbeat, or None
        when nothing was recorded. No clock is read, so downtime and skew are
        never credited as playback; the worst case replays one heartbeat
        interval. `now` feeds only the legacy fallback. Callers still cap at the
        song's duration."""
        if self.last_position_secs is not None and not self._heartbeat_predates_song():
            return max(0, int(self.last_position_secs))
        return self._legacy_wall_clock_position_at(now)

    def _heartbeat_predates_song(self) -> bool:
        """True when the recorded position belongs to an EARLIER song: a build
        without these fields cannot clear them, so `just up <older-sha>` and back
        leaves one song's position parked on a later song's hash. Every
        legitimate write puts the heartbeat at or after the start it belongs to.
        Judged only when both values are present."""
        if self.last_heartbeat_epoch is None or self.play_start_epoch is None:
            return False
        return self.last_heartbeat_epoch < self.play_start_epoch

    def _legacy_wall_clock_position_at(self, now: float) -> int | None:
        """Position extrapolated from the start epoch, for a hash written
        before last_position_secs existed. `now` is read at RESTART, so downtime
        lands on the position. Kept one release: resuming badly beats not
        resuming."""
        if self.play_start_epoch is None:
            return None
        elapsed = now - self.play_start_epoch
        total_pause = self.total_pause_seconds
        if self.pause_start_epoch is not None:
            total_pause += now - self.pause_start_epoch
        return max(0, int(elapsed - total_pause))

    @classmethod
    def from_redis(cls, raw: dict[bytes, bytes]) -> Self:
        """Deserialize raw HGETALL output; an empty dict yields the zero-value
        snapshot."""
        # No `_b_float(...) or 0.0` on total_pause: 0.0 is falsy and a stored
        # 0.0 would be elevated to the default.
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
            current_song_is_resume=(
                _b_str(raw, StateField.CURRENT_SONG_IS_RESUME) == "1"
            ),
            current_song_start_paused=(
                _b_str(raw, StateField.CURRENT_SONG_START_PAUSED) == "1"
            ),
            # `or` coalescing is safe on these: the zero value IS the default.
            current_song_queued_at=(
                _b_float(raw, StateField.CURRENT_SONG_QUEUED_AT) or 0.0
            ),
            current_song_queue_position=(
                _b_opt_int(raw, StateField.CURRENT_SONG_QUEUE_POSITION) or 0
            ),
            current_song_query_source=_b_str(raw, StateField.CURRENT_SONG_QUERY_SOURCE),
            current_song_user_input=(
                _b_str(raw, StateField.CURRENT_SONG_USER_INPUT) or None
            ),
            current_song_played_at=(
                _b_float(raw, StateField.CURRENT_SONG_PLAYED_AT) or 0.0
            ),
            play_start_epoch=_b_float(raw, StateField.PLAY_START_EPOCH),
            total_pause_seconds=total_pause if total_pause is not None else 0.0,
            pause_start_epoch=_b_float(raw, StateField.PAUSE_START_EPOCH),
            last_position_secs=_b_float(raw, StateField.LAST_POSITION_SECS),
            last_heartbeat_epoch=_b_float(raw, StateField.LAST_HEARTBEAT_EPOCH),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class NowPlayingData:
    """Typed snapshot of guild:{id}:now_playing. from_song() builds it from a
    live YTDL for the start-song write, from_redis() rebuilds it during crash
    recovery — one type, so the live and recovered embeds cannot drift."""

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
        """The one field extraction behind both the live embed and the Redis
        snapshot."""
        return cls(
            title=song.title or "",
            webpage_url=song.webpage_url or "",
            uploader=song.uploader or "",
            # Empty for unknown duration (livestream) rather than "0:00": the
            # recovered embed keys its Duration line off this being truthy.
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
        """Deserialize raw HGETALL output; None when the hash is empty (it is
        DELETE'd wholesale on song end, so empty == no song)."""
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
        """Flat string dict for HSET, spelled out (not asdict()) so the wire
        schema is pinned to NowPlayingField rather than attribute names."""
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
    # Interjection flags — absent on pre-feature entries, parsed as False.
    INTERJECTED: Final[str] = "interjected"
    IS_RESUME: Final[str] = "is_resume"
    START_PAUSED: Final[str] = "start_paused"
    # Ask-time analytics, on both entry types. FLAT on the wire although they
    # group as Analytics in memory. Absent on pre-feature entries → 0 defaults.
    QUEUED_AT: Final[str] = "queued_at"
    QUEUE_POSITION: Final[str] = "queue_position"
    # Parse-time classification, on both entry types (see sources.py).
    QUERY_SOURCE: Final[str] = "query_source"
    # When the audio started; "qobj" entries only. Absent → 0.0.
    PLAYED_AT: Final[str] = "played_at"
    # The frozen Now Playing card a resume tail disposes of. Absent → 0/0/False.
    NP_MESSAGE_ID: Final[str] = "np_message_id"
    NP_CHANNEL_ID: Final[str] = "np_channel_id"
    NP_DEDICATED: Final[str] = "np_dedicated"
    # "ytsource" entries
    YTSEARCH: Final[str] = "ytsearch"
    URL: Final[str] = "url"
    PROCESS: Final[str] = "process"


# Wire discriminator values; entries written before and after stay readable.
_ENTRY_TYPE_SONG: Final[str] = "qobj"
_ENTRY_TYPE_SEARCH: Final[str] = "ytsource"


# ── Queue-entry value objects — the guild:{id}:queue list at rest ────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class SongQueueEntry:
    """A resolved song at rest ("qobj" on the wire), the pure-data twin of
    src.youtube.QueueObject. requester is an ID (a live discord.Member cannot
    exist at rest; GuildQueue rehydrates it), None only for the crashed-head
    entry. Snowflakes stay exact end-to-end: orjson native ints, never floats."""

    webpage_url: str
    title: str
    requester_id: int | None
    ts: int | None = None
    user_input: str | None = None
    duration: int | None = None
    uploader: str | None = None
    thumbnail: str | None = None
    persisted: bool = True
    # Interjection flags — see the matching QueueObject field comments.
    interjected: bool = False
    is_resume: bool = False
    start_paused: bool = False
    # Ask-time analytics (0 = unknown / played immediately), see Analytics.
    queued_at: float = 0.0
    queue_position: int = 0
    # How it was asked for ("" = unknown), see QueueObject.
    query_source: str = ""
    # When the audio started (0.0 = not played yet). Carried so a song
    # interrupted by an interjection or recovered from a crash records the start
    # of the play, not of its last fragment.
    played_at: float = 0.0
    # The interrupted fragment's frozen NP card. The live np_host_ref cannot be
    # serialized, so a rehydrated tail can only DELETE a dedicated card.
    np_message_id: int = 0
    np_channel_id: int = 0
    np_dedicated: bool = False

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
            queued_at=item.analytics.queued_at,
            queue_position=item.analytics.queue_position,
            query_source=item.query_source,
            played_at=item.played_at,
            np_message_id=item.np_message_id,
            np_channel_id=item.np_channel_id,
            np_dedicated=item.np_dedicated,
        )

    @classmethod
    def from_song(cls, song: YTDL) -> Self:
        """The queue-entry view of a now-playing song, write-side twin of
        from_crashed_state(): from_song → HSET state → crash →
        from_crashed_state → re-queue."""
        return cls(
            webpage_url=song.webpage_url or "",
            title=song.title or "",
            requester_id=song.requester.id if song.requester else None,
            duration=song.duration_secs or None,
            uploader=song.uploader,
            interjected=song.interjected,
            # These round-trip through the state hash, so a default here is a
            # loss visible only after a crash: a resume tail returns as a fresh
            # song, a paused stack returns playing, -remove loses the origin.
            is_resume=song.is_resume,
            start_paused=song.start_paused,
            user_input=song.user_input,
            queued_at=song.analytics.queued_at,
            queue_position=song.analytics.queue_position,
            query_source=song.query_source,
            played_at=song.played_at,
        )

    @classmethod
    def from_crashed_state(
        cls, state: GuildStateData, *, position: int | None
    ) -> Self | None:
        """The crashed "current song" as a queue entry — the inverse of
        pop_queue_and_start_song(), whose current_song_* fields ARE the entry it
        LPOPed. None when no crashed song is recorded. persisted=False: that
        LPOP already committed, so the loop must not LPOP again. `position` is
        the caller-computed resume offset.

        FIXME: A song interrupted mid-play by the crash is a resume in everything
        but the flag — `ts` holds the interrupt position while is_resume stays
        false, so the loop announces "Starting song at N seconds" and
        _remaining_secs bills the whole duration. Synthesizing the flag from
        `ts > 0` would also move the queue display and the interjection wording,
        so it wants its own change.
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
            interjected=state.current_song_interjected,
            # Losing these reclassifies a resume tail as a fresh song on every
            # restart, and brings a paused stack back playing.
            is_resume=state.current_song_is_resume,
            start_paused=state.current_song_start_paused,
            queued_at=state.current_song_queued_at,
            queue_position=state.current_song_queue_position,
            query_source=state.current_song_query_source,
            # This hash is the only place the origin link survives a restart.
            user_input=state.current_song_user_input,
            # The only at-rest copy of a playing song's start (its queue entry
            # was LPOPed), read back rather than restamped. Absent = 0.0.
            played_at=state.current_song_played_at,
        )

    def to_redis(self) -> bytes:
        """Serialize to the wire format; the table pins the schema to
        QueueEntryField, not to attribute names."""
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
                QueueEntryField.QUEUED_AT: self.queued_at,
                QueueEntryField.QUEUE_POSITION: self.queue_position,
                QueueEntryField.QUERY_SOURCE: self.query_source,
                QueueEntryField.PLAYED_AT: self.played_at,
                QueueEntryField.NP_MESSAGE_ID: self.np_message_id,
                QueueEntryField.NP_CHANNEL_ID: self.np_channel_id,
                QueueEntryField.NP_DEDICATED: self.np_dedicated,
            }
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SearchQueueEntry:
    """An unresolved search at rest ("ytsource" on the wire) — e.g. a Spotify
    playlist track awaiting yt-dlp resolution. Holds exactly the YTSource fields
    the wire persists; the rest default on rehydration."""

    ytsearch: str | None = None
    url: str | None = None
    process: bool | None = None
    ts: int | None = None
    # What the user typed, for -remove: the ytsearch here is a generated title.
    user_input: str | None = None
    # Ask-time analytics, so a Spotify playlist track keeps its position through
    # the resolve at dequeue.
    queued_at: float = 0.0
    queue_position: int = 0
    # The leg that makes a Spotify playlist track archive as Spotify rather than
    # as the YouTube URL it becomes.
    query_source: str = ""

    @classmethod
    def from_ytsource(cls, source: YTSource) -> Self:
        return cls(
            ytsearch=source.ytsearch,
            url=source.url,
            process=source.process,
            ts=source.ts,
            user_input=source.user_input,
            queued_at=source.analytics.queued_at,
            queue_position=source.analytics.queue_position,
            query_source=source.query_source,
        )

    def to_redis(self) -> bytes:
        return orjson.dumps(
            {
                QueueEntryField.TYPE: _ENTRY_TYPE_SEARCH,
                QueueEntryField.YTSEARCH: self.ytsearch,
                QueueEntryField.URL: self.url,
                QueueEntryField.PROCESS: self.process,
                QueueEntryField.TS: self.ts,
                QueueEntryField.USER_INPUT: self.user_input,
                QueueEntryField.QUEUED_AT: self.queued_at,
                QueueEntryField.QUEUE_POSITION: self.queue_position,
                QueueEntryField.QUERY_SOURCE: self.query_source,
            }
        )


QueueEntry = Union[SongQueueEntry, SearchQueueEntry]


# `bytes | str` matches orjson.loads() and redis-py's declared LRANGE return;
# narrowing to bytes forces a cast at every caller (parse_history_entry likewise).
def parse_queue_entry(data: bytes | str) -> QueueEntry | None:
    """Deserialize one queue-list entry; "type" discriminates searches from
    songs. Corrupt entries return None with a warning, so the rest of the queue
    survives."""
    try:
        d = orjson.loads(data)
        if d.get(QueueEntryField.TYPE) == _ENTRY_TYPE_SEARCH:
            return SearchQueueEntry(
                ytsearch=d.get(QueueEntryField.YTSEARCH),
                url=d.get(QueueEntryField.URL),
                process=d.get(QueueEntryField.PROCESS),
                ts=d.get(QueueEntryField.TS),
                user_input=d.get(QueueEntryField.USER_INPUT),
                queued_at=d.get(QueueEntryField.QUEUED_AT, 0.0),
                queue_position=d.get(QueueEntryField.QUEUE_POSITION, 0),
                query_source=d.get(QueueEntryField.QUERY_SOURCE, ""),
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
            queued_at=d.get(QueueEntryField.QUEUED_AT, 0.0),
            queue_position=d.get(QueueEntryField.QUEUE_POSITION, 0),
            query_source=d.get(QueueEntryField.QUERY_SOURCE, ""),
            played_at=d.get(QueueEntryField.PLAYED_AT, 0.0),
            np_message_id=d.get(QueueEntryField.NP_MESSAGE_ID, 0),
            np_channel_id=d.get(QueueEntryField.NP_CHANNEL_ID, 0),
            np_dedicated=d.get(QueueEntryField.NP_DEDICATED, False),
        )
    except Exception as e:
        log.warning(f"guild_state: corrupt queue entry dropped: {e}")
        return None


# ── guild:{id}:history list — wire format ────────────────────────────────────
# One JSON object of HistoryEntryField keys per entry, most-recently-recorded
# first (song-end order; GuildHistory.recent sorts on played_at).


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
    CHANNEL_ID: Final[str] = "channel_id"
    QUEUED_AT: Final[str] = "queued_at"
    QUEUE_POSITION: Final[str] = "queue_position"
    QUERY_SOURCE: Final[str] = "query_source"


# The play_history column domain (migrations/0001_play_history.sql), kept beside
# the dataclass that guarantees it so the type and its domain cannot drift.
_TEXT_FIELDS: Final[tuple[str, ...]] = (
    "title",
    "webpage_url",
    "requester_name",
    "thumbnail",
    "uploader",
)
_INT4_FIELDS: Final[tuple[str, ...]] = (
    "duration_secs",
    "played_secs",
    "queue_position",
)
_INT8_FIELDS: Final[tuple[str, ...]] = (
    "guild_id",
    "requester_id",
    "message_id",
    "channel_id",
)
# Machine-minted tokens (src.sources.query_source_of): a lowercase host, or the
# literal "search". Anything else is a producer defect and clamps to the unknown
# sentinel, so play_history's CHECK on the same domain never fires.
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9.-]{0,64}")
_SLUG_FIELDS: Final[tuple[str, ...]] = ("query_source",)
# The timestamptz columns: clamped to the epoch sentinel rather than to a bound,
# since a value outside the range is a corrupt clock, not a large one.
_EPOCH_FIELDS: Final[tuple[str, ...]] = ("played_at", "queued_at")
_INT4_MAX: Final[int] = 2**31 - 1
_INT8_MAX: Final[int] = 2**63 - 1
# 9999-12-31T23:59:59Z — the epoch domain of the play_history timestamptz
# columns: history_archive clamps cutoffs to it and the migration's CHECKs
# spell the same value.
TS_MAX: Final[float] = 253402300799.0


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryEntry:
    """One played song at rest — an element of guild:{id}:history, matching the
    play_history row. Zero-values mean "unknown": absent wire fields default on
    parse and the display layer degrades.

    guild_id is redundant on the per-guild list but required on the global
    history:outbox stream, where the drainer maps each entry to a row; entries
    written before the field existed parse as guild_id=0.

    message_id and channel_id are a WEAK reference, taken off the same message
    at song end so the pair is both real or both 0: the NP host migrates across
    messages during one song and a dedicated host is deleted when retired, so it
    is neither a foreign key nor part of play_history_dedup.
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
    # Unix epoch when the audio started; drives <t:…:f>. One value per play, not
    # per fragment: an interjection's resume tail inherits the interrupted song's
    # stamp.
    played_at: float = 0.0
    message_id: int = 0  # NP host at song end; 0 = unknown (see class docstring)
    channel_id: int = 0  # the channel that host was in; 0 = unknown, always paired
    queued_at: float = 0.0  # unix epoch when the user ASKED; 0 = unknown
    # Songs ahead at ask time, counting the one playing. 0 = played immediately,
    # also what a pre-feature entry parses as.
    queue_position: int = 0
    # "search", or the host of the pasted link; "" = unknown. Classified at parse
    # time (src.sources) because a Spotify link and a plaintext search both
    # resolve to a YouTube watch URL.
    query_source: str = ""

    def __post_init__(self) -> None:
        """Normalize into the play_history column domain: an instance is, by
        construction, a row Postgres accepts, and every producer routes through
        here. Total — never raises — because it runs on the read path over rows
        that predate it; strictness lives in the DB CHECK constraints. No type
        coercion: HistoryEntry(title=None) raising TypeError here is correct.

        One exemption: integers floor at 0, but play_history's CHECK on guild_id
        is `> 0`, so a guild_id-0 entry is constructible and not insertable.
        Refusing at the database routes it to play_history_rejected, where a row
        means a producer stopped stamping guild_id.

        Range tests rather than paired bound checks: a chained comparison is
        False for NaN, which lands NaN on the sentinel. Everything else the
        columns could refuse (lone surrogates, non-finite floats, >64-bit ints)
        orjson already refuses to encode.
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
        for name in _EPOCH_FIELDS:
            value_epoch: float = getattr(self, name)
            if not 0.0 <= value_epoch <= TS_MAX:
                object.__setattr__(self, name, 0.0)
        for name in _SLUG_FIELDS:
            value_slug: str = getattr(self, name)
            if _SLUG_RE.fullmatch(value_slug) is None:
                object.__setattr__(self, name, "")

    @classmethod
    def from_song(
        cls, song: YTDL, *, guild_id: int, message_id: int, channel_id: int
    ) -> Self:
        """Extraction from a finished song. The ids are keyword-required because
        a forgotten stamp writes cleanly as 0, indistinguishable from a song that
        had no host — pass 0 explicitly for that, both ids off the same message.
        played_at rides the song (stamped at vc.play(), inherited by every later
        fragment). played_secs is the position reached, capped at duration when
        known; an interrupted song is recorded once at its resume tail."""
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
            played_at=song.played_at,
            message_id=message_id,
            channel_id=channel_id,
            queued_at=song.analytics.queued_at,
            queue_position=song.analytics.queue_position,
            query_source=song.query_source,
        )

    @classmethod
    def from_queue_object(cls, item: QueueObject, *, guild_id: int) -> Self:
        """A played song recorded as it LEAVES the queue — the -clear/-remove
        counterpart to from_song, for an interjection-interrupted entry destroyed
        before its tail could play. played_secs comes from `ts`, the ABSOLUTE
        resume offset, capped at duration. The host ids come off the tail's
        np_* fields: the cleanup that deletes that card fires only when a tail
        STARTS, and a flushed tail never does."""
        played = item.ts or 0
        duration = item.duration or 0
        if duration:
            played = min(played, duration)
        return cls(
            guild_id=guild_id,
            title=item.title,
            webpage_url=item.webpage_url,
            duration_secs=duration,
            played_secs=played,
            requester_id=item.requester.id if item.requester else 0,
            requester_name=item.requester.display_name if item.requester else "",
            thumbnail=item.thumbnail or "",
            uploader=item.uploader or "",
            played_at=item.played_at,
            message_id=item.np_message_id,
            channel_id=item.np_channel_id,
            queued_at=item.analytics.queued_at,
            queue_position=item.analytics.queue_position,
            query_source=item.query_source,
        )

    def to_redis(self) -> bytes:
        """Serialize to the wire format; the table pins the schema to
        HistoryEntryField, not to attribute names."""
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
                HistoryEntryField.CHANNEL_ID: self.channel_id,
                HistoryEntryField.QUEUED_AT: self.queued_at,
                HistoryEntryField.QUEUE_POSITION: self.queue_position,
                HistoryEntryField.QUERY_SOURCE: self.query_source,
            }
        )


def serialize_history_entry(entry: HistoryEntry) -> bytes:
    return entry.to_redis()


def parse_history_entry(data: bytes | str) -> HistoryEntry | None:
    """Deserialize one history-list entry; corrupt entries are dropped with a
    warning. Unknown keys are ignored and missing keys default, so mixed-build
    readers stay tolerant in both directions."""
    try:
        entry = orjson.loads(data)
        if not isinstance(entry, dict):
            log.warning(
                f"guild_state: corrupt history entry dropped: unexpected JSON type ({type(entry).__name__})"
            )
            return None
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
            channel_id=int(entry.get(HistoryEntryField.CHANNEL_ID) or 0),
            queued_at=float(entry.get(HistoryEntryField.QUEUED_AT) or 0.0),
            queue_position=int(entry.get(HistoryEntryField.QUEUE_POSITION) or 0),
            query_source=str(entry.get(HistoryEntryField.QUERY_SOURCE) or ""),
        )
    except Exception as e:
        log.warning(f"guild_state: corrupt history entry dropped: {e}")
        return None


# ── The aggregate — a guild's persisted playback state, read as one unit ─────


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildPlaybackSnapshot:
    """A guild's complete persisted playback aggregate — state hash, pending
    queue, now-playing snapshot, history, config — read in one pipelined round
    trip (GuildRedisStore.get_playback_snapshot). Recovery decisions spanning
    the halves live here as named properties."""

    state: GuildStateData
    queue: tuple[QueueEntry, ...] = ()
    # None when no song was playing (empty hash == no song).
    now_playing: NowPlayingData | None = None
    # Newest-first, as stored (GuildHistory.restore() reverses).
    history: tuple[HistoryEntry, ...] = ()
    # Read in the same round trip because restore is when they are needed.
    config: GuildConfig = GuildConfig()

    @property
    def stored_volume(self) -> float | None:
        """The guild's volume, or None if it never set one. Config first, then
        the legacy state field (one-release migration path: restore SEEDS config
        from it via migrate_volume, HSETNX, and set_volume keeps both fresh, so
        a rollback is a no-op). Delete this leg with StateField.VOLUME."""
        return (
            self.config.volume if self.config.volume is not None else self.state.volume
        )

    @property
    def pending_count(self) -> int:
        return len(self.queue)

    @property
    def has_restorable_playback(self) -> bool:
        """The restore_guild() gate: pending queue entries or a song mid-play at
        crash time."""
        return bool(self.queue) or self.state.has_crashed_song


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildRecoveryGate:
    """The minimal read restore_guild() needs to decide whether to reconnect:
    the state hash plus the pending queue's LENGTH, never its contents, so a
    -stopped guild's leftover queue stays off the wire on every on_ready."""

    state: GuildStateData
    pending_count: int = 0

    @property
    def has_restorable_playback(self) -> bool:
        """GuildPlaybackSnapshot.has_restorable_playback over the queue length."""
        return self.pending_count > 0 or self.state.has_crashed_song


# ── -analytics: the render worker's boundary payload ──────────────────────────
# Unpickling a dataclass imports its defining module, so their home decides what
# a chart worker drags in — this module is stdlib + orjson. Every field
# defaults, so an entry written by an older build still decodes.
# See docs/ARCHITECTURE.md#analytics-rendering.


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyPoint:
    """One x-position of the per-day series. `day` is an ISO date naming the
    bucket's first UTC day — at --days 365 the bucket is a week, and the date
    is its Monday."""

    day: str = ""
    plays: int = 0
    listen_secs: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceDay:
    """One segment of a stacked bar: plays a query_source contributed to one
    bucket. Sparse — a source with no plays that day has no row."""

    day: str = ""
    source: str = ""
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class HeatCell:
    """One weekday x hour-of-day cell. `dow` is ISO (1 = Monday), `hour` 0-23,
    both UTC — the frame play_history was written in."""

    dow: int = 0
    hour: int = 0
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class CompletionBucket:
    """Plays whose played/duration ratio fell in one tenth, per source. `bucket`
    is 1-10 covering [0,0.1) .. [0.9,1.0]; the SQL folds width_bucket's overflow
    11 into 10 because a ratio of exactly 1.0 is the modal value."""

    source: str = ""
    bucket: int = 0
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class DurationBucket:
    """Plays by song length in whole minutes, 0-19, with 20 meaning "20 min or
    longer" — an open top bucket, so an hour-long mix does not stretch the
    axis."""

    minutes: int = 0
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceCompletion:
    """DURATION-WEIGHTED completion for one source: the sums, not the ratio, so
    the renderer divides once and the zero case is visible. A different
    question from CompletionBucket — a handful of abandoned hour-long mixes
    dominate the weighted number."""

    source: str = ""
    played_secs: int = 0
    duration_secs: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TopListener:
    """One row of the embed's listeners section. Distinct from history_archive's
    RequesterLeader: this one must not live in a module the render worker would
    have to import."""

    requester_id: int = 0
    requester_name: str = ""
    plays: int = 0
    played_secs: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TopArtist:
    uploader: str = ""
    plays: int = 0
    played_secs: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TopSong:
    title: str = ""
    webpage_url: str = ""
    query_source: str = ""
    plays: int = 0
    played_secs: int = 0


# "No usable queue-wait data in this window". Negative because 0 is a legitimate
# wait (a song queued into an empty queue).
WAIT_UNAVAILABLE: Final[float] = -1.0
# The queue-wait percentiles and the median's index. The SQL array, the length
# checks and the figure's labels all read from here.
WAIT_PERCENTILES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 0.9)
WAIT_MEDIAN_INDEX: Final[int] = WAIT_PERCENTILES.index(0.5)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyticsMetrics:
    """Everything -analytics knows about one guild and one window. The whole
    object crosses to the chart worker, including the three top-N lists it
    never draws (~2 KB of ~15 KB). The worker may NOT draw any of their strings:
    the runtime image has no CJK, Thai or emoji glyphs, so a human-authored name
    renders as tofu (_ascii_safe() in analytics_render enforces it). `days` is
    the REQUESTED window and archived_days its real coverage; the title names
    both when they differ."""

    days: int = 0
    # Both epochs, so the footer and the cache TTL read the same clock the
    # buckets did. window_end is exclusive: the day (or week) boundary the
    # window stops at, never "now".
    window_start_epoch: float = 0.0
    window_end_epoch: float = 0.0
    # "day" or "week". At 365 days the SQL downsamples, and the y-axis label
    # has to move with it.
    bucket_unit: str = "day"
    # Today's UTC midnight, from the same clock read as the bounds; the cache
    # expiry derives from it.
    today_start_epoch: float = 0.0
    archived_days: int = 0

    plays: int = 0
    listen_secs: int = 0
    unique_songs: int = 0
    unique_listeners: int = 0
    unique_artists: int = 0
    wait_p50_secs: float = WAIT_UNAVAILABLE
    # Plays with duration_secs <= 0 (livestreams): counted everywhere else,
    # excluded from the completion panel alone, which names the number.
    livestream_plays: int = 0

    daily: tuple[DailyPoint, ...] = ()
    daily_by_source: tuple[SourceDay, ...] = ()
    heat: tuple[HeatCell, ...] = ()
    completion: tuple[CompletionBucket, ...] = ()
    durations: tuple[DurationBucket, ...] = ()
    source_completion: tuple[SourceCompletion, ...] = ()
    # p10/p25/p50/p75/p90 of queue wait, seconds. Empty when no row in the
    # window had a non-sentinel queued_at.
    wait_pcts: tuple[float, ...] = ()

    top_listeners: tuple[TopListener, ...] = ()
    top_artists: tuple[TopArtist, ...] = ()
    top_songs: tuple[TopSong, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Reads the explicit play count, never a branch's shape: every
        aggregate coalesces to `[]`, so an empty daily series also describes a
        guild whose rows all failed one branch's filter."""
        return self.plays == 0

    @property
    def period_label(self) -> str:
        """The window as a period, in the unit it is bucketed by. 365 mod 7 is
        1, so the weekly window covers 53 whole weeks (371 days), and calling
        that "last 365 days" names six fewer days than the chart draws. Lives
        here because the card and the figure must agree and the figure cannot
        import the card."""
        if self.bucket_unit == "week":
            weeks = -(-self.days // 7)
            return f"last {weeks} week{'s' if weeks != 1 else ''}"
        return f"last {self.days} day{'s' if self.days != 1 else ''}"
