"""
Guild state schema — single source of truth for all Redis state stored per guild.

The two Redis hashes and the queue list each have corresponding frozen dataclasses
(value objects). GuildRedisStore in redis_client.py uses these for typed reads and
field-name constants; GuildQueue in guild_queue.py converts between at-rest queue
entries and live queue items. Callers never touch raw bytes from Redis directly.

Pure schema: constructors, serializers, derived read-only properties, and the
domain NORMALIZATION that keeps a value object inside the column domain it is
stored in (HistoryEntry.__post_init__) — no behaviour, and no runtime imports from
the rest of the project (orjson is the project-wide wire codec). Normalization
lives here so the type and the domain it promises cannot drift apart.
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
    storage alone: read only to serialize or to carry onto the next object.
    Admission rule — a field anything branches on or renders belongs elsewhere.

    An in-memory shape only. Wire entries and play_history columns stay FLAT,
    exploded at the serialization boundary in this module.

    Frozen: carry sites alias one instance across a resume tail and its source
    (analytics=current.analytics)."""

    # Unix epoch when the user ASKED for the song: the command message's
    # snowflake time. Discord's clock, so played_at - queued_at is cross-clock
    # and goes slightly negative under host drift.
    # 0.0 = unknown (pre-feature wire entries).
    queued_at: float
    # Songs ahead at ask time, counting the one playing (0 = played immediately).
    # Read once at command dispatch (MusicPlayer.enqueue_depth), so the loop's
    # continuous dequeuing leaves it approximate against the insert.
    queue_position: int


# The unknown value: what a pre-feature wire entry rehydrates as, and the
# default on live objects whose construction site cannot know the values yet.
ANALYTICS_ZERO: Final[Analytics] = Analytics(queued_at=0.0, queue_position=0)


# ── guild:{id}:state hash — field name constants ─────────────────────────────


class StateField:
    # LEGACY. Volume moved to guild:{id}:config (GuildConfig) because this hash
    # expires in 24h and a setting must not. Config is the source of truth and is
    # read first; this field is still both read AND written for one release, so a
    # rollback to a build that only knows this one still finds a fresh value —
    # deleting it outright reset every migrated guild to 100% on `just up
    # <older-sha>`. Drop this, GuildStateData.volume and set_volume's second write
    # together once every deployment understands :config.
    VOLUME: Final[str] = "volume"
    VOICE_CHANNEL_ID: Final[str] = "voice_channel_id"
    TEXT_CHANNEL_ID: Final[str] = "text_channel_id"
    CURRENT_SONG_URL: Final[str] = "current_song_url"
    CURRENT_SONG_TITLE: Final[str] = "current_song_title"
    CURRENT_SONG_DURATION: Final[str] = "current_song_duration"
    CURRENT_SONG_UPLOADER: Final[str] = "current_song_uploader"
    CURRENT_SONG_REQUESTER_ID: Final[str] = "current_song_requester_id"
    # "1" when the playing song was queued by an interjection — attribution that
    # would otherwise be lost across a crash mid-interjection.
    CURRENT_SONG_INTERJECTED: Final[str] = "current_song_interjected"
    # "1" when the playing song is an interjection's resume tail, and when parked
    # paused. is_resume drives the resume announcement, _remaining_secs' billing and
    # the tail's NP-card cleanup, so losing it across a crash changes behaviour.
    CURRENT_SONG_IS_RESUME: Final[str] = "current_song_is_resume"
    CURRENT_SONG_START_PAUSED: Final[str] = "current_song_start_paused"
    # Set once at ask time and carried, never rewritten, so a crash-recovered
    # song still archives the position it was originally queued at.
    CURRENT_SONG_QUEUED_AT: Final[str] = "current_song_queued_at"
    CURRENT_SONG_QUEUE_POSITION: Final[str] = "current_song_queue_position"
    # Same reason: without it a song recovered after a crash archives as unknown,
    # silently and only for crashed plays.
    CURRENT_SONG_QUERY_SOURCE: Final[str] = "current_song_query_source"
    # What the user typed. Carried for -remove rather than the archive: without it
    # a crash-recovered head is the one track a collection link cannot take out.
    CURRENT_SONG_USER_INPUT: Final[str] = "current_song_user_input"
    # When the audio started. Not PLAY_START_EPOCH below, which is backdated by the
    # -ss offset, and not derivable from this run's clock at all — a resume tail
    # inherits its value from an earlier fragment.
    CURRENT_SONG_PLAYED_AT: Final[str] = "current_song_played_at"
    # LEGACY, all three: superseded by LAST_POSITION_SECS below and read only by
    # _legacy_wall_clock_position_at. Still written so a rollback recovers; drop them
    # with that method and on_pause/on_resume one release after the heartbeat ships.
    PLAY_START_EPOCH: Final[str] = "play_start_epoch"
    TOTAL_PAUSE_SECONDS: Final[str] = "total_pause_seconds"
    PAUSE_START_EPOCH: Final[str] = "pause_start_epoch"
    # The recorded playback position, read with no wall-clock arithmetic. The
    # three fields above are its predecessors, still written for rollback safety;
    # they go one release after this ships.
    LAST_POSITION_SECS: Final[str] = "last_position_secs"
    # When that position was recorded. Never an addend — it dates the field above,
    # so _heartbeat_predates_song can refuse one belonging to an earlier song.
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


# ── Parsing helpers (module-level; shared by both from_redis constructors) ───
#
# The bare `except A, B:` clauses below are PEP 758 (Python 3.14+) multi-exception
# syntax, not the old Python-2 catch-one/bind-name form: they catch the tuple
# (A, B) and let unlisted types propagate. ruff's formatter normalizes to this at
# `target-version = "py314"`; do not re-parenthesize, ruff strips it back.


def _b_str(raw: dict[bytes, bytes], key: str, default: str = "") -> str:
    v = raw.get(key.encode())
    # `is None`, not truthiness: a missing key gets the default, a stored b"" stays
    # "". errors="replace" so one corrupt non-UTF8 byte degrades to a mangled
    # string — a strict decode would raise out of from_redis(), get_guild_state()
    # would return None, and corruption would be misclassified as "Redis
    # unavailable", blocking recovery until TTL expiry.
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


class ConfigField:
    """Wire field names for guild:{id}:config. Spelled out, like every other field
    table here, so renaming a Python attribute can never silently rename a Redis
    field and orphan every guild's stored setting."""

    DEBUG_MODE: Final[str] = "debug_mode"
    VOLUME: Final[str] = "volume"
    TIMEZONE: Final[str] = "timezone"


# The zone every guild renders ETAs in until it picks one. Named here rather than
# in musicplayer so the schema layer can validate against the same default it
# hands back, and so the eventual `-options timezone <name>` has one answer to
# "what does unset mean?".
DEFAULT_TIMEZONE: Final[str] = "America/Los_Angeles"

# Zone names already proven unusable on this host. ZoneInfo caches SUCCESSFUL
# lookups itself; a failed one is the expensive direction (a filesystem miss, ~155us
# measured) and it also logs, so without this a stored name the host cannot resolve
# would pay both on every render once -options lets a guild set one. Bounded by the
# cap rather than by trust: the write boundary validates, but this key outlives
# builds and can be hand-edited.
_UNUSABLE_ZONES: Final[set[str]] = set()
_MAX_UNUSABLE_ZONES: Final[int] = 256

# The longest real IANA name is 32 chars ("America/Argentina/ComodRivadavia").
_MAX_TIMEZONE_NAME: Final[int] = 64


@lru_cache(maxsize=1)
def _known_zones() -> frozenset[str]:
    """Every zone this host can name. Cached: available_timezones() walks the whole
    tz database, and the answer cannot change without a restart."""
    return frozenset(available_timezones())


def valid_timezone(name: str) -> bool:
    """True if `name` is a zone this host can actually resolve.

    The WRITE boundary's check. tzinfo()'s fallback is a backstop for a name that
    stopped resolving (a base-image change), not a substitute for this: a bad name
    stored unvalidated fails SILENTLY — the write succeeds, the command reports
    success, and the guild's ETAs stay on the default forever with only a log line
    nobody reads.

    Membership, not `ZoneInfo(name)`: ZoneInfo resolves against the tz database BY
    PATH, so `zone.tab` and `leapseconds` are real files that construct fine without
    being zones.
    """
    if not name or len(name) > _MAX_TIMEZONE_NAME:
        return False
    return name in _known_zones()


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildConfig:
    """A guild's DURABLE preferences — what an operator chose, not what the bot is
    doing right now.

    Deliberately its own key rather than fields on guild:{id}:state. That hash is
    runtime state (current song, pause epochs) and carries a 24h TTL, so a setting
    stored there would silently revert on any guild that went a day without playing
    anything — a worse failure than the in-memory version it replaces, because the
    reset would be tied to nothing the user can see.

    Every field is Optional and that is the whole point: absent means "follow the
    host default", which is NOT the same as an explicitly chosen value. A guild that
    turned debug off while the host default is on must stay off, and a guild that
    never touched it must follow the host — a plain bool cannot express both. Volume
    carries the same distinction for the same reason GuildStateData.volume did:
    restore must skip the assignment rather than clobber a concurrent -volume with a
    fabricated 1.0.
    """

    debug_mode: bool | None = None
    volume: float | None = None
    # An IANA name, not a ZoneInfo: the wire stays human-readable and the
    # eventual -options command can write what the user typed. Resolved by
    # tzinfo() below, which is also where an unusable name degrades.
    timezone: str | None = None

    def to_redis(self) -> dict[str, str]:
        """Only fields with a value. An unset field is ABSENT from the hash rather
        than stored as a sentinel, so "never chose" stays distinguishable after a
        round trip."""
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

        Resolution happens HERE rather than at write time because the tz database is
        a property of the host, not of the stored value: a name that resolved when it
        was set can stop resolving after a base-image change, and an ETA rendered in
        the default beats a render path that raises.
        """
        # tzdata is a declared dependency, so the default always resolves and this
        # cannot raise on a render path even where the host ships no tz database.
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
        """Deserialize raw HGETALL output; an empty dict yields all-unset.

        Anything that is neither "1" nor "0" reads as unset rather than raising:
        this key outlives builds, and one unparseable field must not cost the guild
        its whole config (the same rule the queue and history parsers follow).
        """
        stored = _b_str(raw, ConfigField.DEBUG_MODE)
        debug_mode = {"1": True, "0": False}.get(stored)
        # _b_float already logs and returns None on a malformed value, which is the
        # same "unset" this class wants.
        return cls(
            debug_mode=debug_mode,
            volume=_b_float(raw, ConfigField.VOLUME),
            timezone=_b_str(raw, ConfigField.TIMEZONE) or None,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildStateData:
    """Typed snapshot of guild:{id}:state deserialized from Redis.

    Zero-value defaults throughout, so GuildStateData() is the canonical "empty
    hash" snapshot. volume is None when nothing is stored, not 1.0: the caller must
    tell "nothing persisted" from "user set 1.0" so a restore can skip the
    assignment instead of clobbering a concurrent -volume with a fabricated default.
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

        Named for the crash, not is_paused, to stay distinct from live state
        (vc.is_paused()): this is persisted crash-time state only."""
        return self.pause_start_epoch is not None

    def crashed_position_at(self, now: float) -> int | None:
        """Playback position (seconds) at the last recorded heartbeat, or None
        when nothing was recorded.

        No clock is read, so neither downtime nor skew between restarts can be
        credited as playback. Resumes at last_position_secs exactly, so the worst
        case replays one heartbeat interval — the deliberate bias, since replaying
        3s is imperceptible and skipping 3s is not. `now` feeds only the legacy
        fallback. Callers still cap at the song's duration to stop an EOF seek.
        """
        if self.last_position_secs is not None and not self._heartbeat_predates_song():
            return max(0, int(self.last_position_secs))
        return self._legacy_wall_clock_position_at(now)

    def _heartbeat_predates_song(self) -> bool:
        """True when the recorded position belongs to an EARLIER song.

        A build without these fields cannot clear them, so `just up <older-sha>`
        and back leaves one song's position parked on a later song's hash — which
        would resume it minutes in. Every legitimate write puts the heartbeat at or
        after the start it belongs to: the seed writes play_start_epoch +
        start_offset, and ticks read a clock the backdated epoch precedes. Judged
        only when both values are present, so a corrupt one costs nothing.
        """
        if self.last_heartbeat_epoch is None or self.play_start_epoch is None:
            return False
        return self.last_heartbeat_epoch < self.play_start_epoch

    def _legacy_wall_clock_position_at(self, now: float) -> int | None:
        """Pre-heartbeat position math: extrapolate from the start epoch.

        Reads a state hash written before last_position_secs existed. `now` is read
        at RESTART, so downtime lands straight on the position — the bug the
        heartbeat replaces. Kept one release: resuming badly beats not resuming.
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
            current_song_is_resume=(
                _b_str(raw, StateField.CURRENT_SONG_IS_RESUME) == "1"
            ),
            current_song_start_paused=(
                _b_str(raw, StateField.CURRENT_SONG_START_PAUSED) == "1"
            ),
            # `or` coalescing is safe on these two: the zero value IS the default,
            # unlike total_pause_seconds below.
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
    """Typed snapshot of guild:{id}:now_playing.

    Bidirectional: from_song() builds it from a live YTDL for the atomic start-song
    write, from_redis() rebuilds it during crash recovery. One type, so the live
    and recovered embeds can't drift."""

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
            # Empty for unknown duration (livestream, missing metadata) rather than
            # fmt_duration's "0:00" — mirrors the live embed, which draws no bar at
            # duration_secs <= 0, and the recovered embed keys its Duration line
            # off this being truthy.
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
        schema to Python attribute names — renaming one would silently rename the
        hash field. This table pins it to the NowPlayingField constants.
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
    # Interjection flags — absent on pre-feature entries, parsed as False.
    INTERJECTED: Final[str] = "interjected"
    IS_RESUME: Final[str] = "is_resume"
    START_PAUSED: Final[str] = "start_paused"
    # Ask-time analytics, carried on both entry types. FLAT on the wire even
    # though they group as Analytics in memory — nesting would break these
    # parsers, the clamp-domain coverage and the positional row mapping. Absent
    # on pre-feature entries, parsed as the 0 defaults.
    QUEUED_AT: Final[str] = "queued_at"
    QUEUE_POSITION: Final[str] = "queue_position"
    # Parse-time classification, carried on both entry types (see sources.py).
    QUERY_SOURCE: Final[str] = "query_source"
    # When the audio started. Only "qobj" entries carry it — a search has not
    # played by definition. Absent on pre-feature entries, parsed as 0.0.
    PLAYED_AT: Final[str] = "played_at"
    # The frozen Now Playing card a resume tail is responsible for disposing of.
    # Absent on pre-feature entries, parsed as 0 / 0 / False = "nothing to clean".
    NP_MESSAGE_ID: Final[str] = "np_message_id"
    NP_CHANNEL_ID: Final[str] = "np_channel_id"
    NP_DEDICATED: Final[str] = "np_dedicated"
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
    requester_id is None only for the crashed-head entry from from_crashed_state();
    snowflakes stay exact end-to-end (orjson native ints, never a float path).
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
    # Interjection flags — see the matching QueueObject field comments.
    interjected: bool = False
    is_resume: bool = False
    start_paused: bool = False
    # Ask-time analytics (0 = unknown / played immediately), see Analytics.
    queued_at: float = 0.0
    queue_position: int = 0
    # How it was asked for ("" = unknown), see QueueObject.
    query_source: str = ""
    # When the audio started (0.0 = not played yet), see QueueObject. Carried so a
    # song interrupted by an interjection, or recovered from a crash, still records
    # start of the play rather than the start of its last fragment.
    played_at: float = 0.0
    # The interrupted fragment's frozen NP card, see QueueObject. The live
    # np_host_ref beside them cannot be serialized, so a rehydrated tail can only
    # DELETE a dedicated card, never strip-edit a command response.
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
            # These three round-trip through the state hash and back out of
            # from_crashed_state(), so a default here is a loss visible only after
            # a crash: a resume tail returns as a fresh song billing the whole
            # duration, a paused stack returns playing, and -remove loses the origin.
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
        """The crashed "current song" reborn as a queue entry — the typed inverse of
        pop_queue_and_start_song(), whose current_song_* fields ARE the entry it
        LPOPed. None when no crashed song is recorded.

        persisted=False: that LPOP already committed, so this entry is not on the
        Redis list and the loop must not LPOP again for it. `position` is the
        caller-computed resume offset (crashed_position_at() plus its duration cap),
        passed in so this stays a pure field mapping.

        FIXME: A song interrupted mid-play by the crash is a resume in everything
        but the flag — `ts` holds the interrupt position while is_resume stays
        false, so the loop announces "Starting song at 137 seconds" rather than
        resuming, and _remaining_secs bills the whole duration instead of the
        tail, skewing every ETA behind it. A song that WAS a `-play --now` tail is
        fine: from_song() carries is_resume through the hash. Synthesizing the
        flag from `ts > 0` would also move the queue display and the interjection
        wording, so it wants its own change.
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
            # Attribution only, carried so a crash mid-interjection does not
            # silently reclassify how the song was queued.
            interjected=state.current_song_interjected,
            # Losing these reclassifies a resume tail as a fresh song on every
            # restart: no resume announcement, _remaining_secs billing the whole
            # duration, no NP-card cleanup, and a paused stack coming back playing.
            is_resume=state.current_song_is_resume,
            start_paused=state.current_song_start_paused,
            queued_at=state.current_song_queued_at,
            queue_position=state.current_song_queue_position,
            query_source=state.current_song_query_source,
            # Origin, so -remove <what was typed> still matches the recovered song:
            # this hash is the only place the link survives a restart.
            user_input=state.current_song_user_input,
            # The parked entry is the only at-rest copy of a playing song's start
            # (its queue entry was LPOPed when it started), so recovery reads it
            # back rather than restamping to the recovery clock. Absent = 0.0.
            played_at=state.current_song_played_at,
        )

    def to_redis(self) -> bytes:
        """Serialize to the wire format. Field table spelled out so the wire schema
        is pinned to QueueEntryField, not to Python attribute names."""
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
    # What the user typed, for -remove to match on. Nothing downstream can
    # reconstruct it: the ytsearch here is a title the expansion generated.
    user_input: str | None = None
    # Ask-time analytics, carried so a Spotify playlist track keeps its position
    # through the resolve at dequeue.
    queued_at: float = 0.0
    queue_position: int = 0
    # Likewise for the classification — this is the leg that makes a Spotify
    # playlist track archive as Spotify and not as the YouTube URL it becomes.
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
    """Deserialize one queue-list entry into a value object.

    The "type" field discriminates searches from songs. Corrupt entries (bad JSON,
    missing required fields) return None with a warning, so one bad entry is
    dropped and the rest of the queue survives.
    """
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
# first (song-end order — see GuildHistory.recent, which sorts on played_at).


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


# The play_history column domain (migrations/0001_play_history.sql), mirrored here
# because HistoryEntry is what guarantees it — next to the dataclass rather than in
# history_archive.py so the type and its domain cannot drift apart, and so this
# module stays free of runtime imports from the rest of the project.
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
# Machine-generated tokens (src.sources.query_source_of), never raw user text: a
# lowercase host, or the literal "search". Anything else is a producer defect, so
# it clamps to the unknown sentinel rather than being stored — which is what lets
# play_history CHECK the same domain without the constraint ever firing.
_SLUG_RE: Final[re.Pattern[str]] = re.compile(r"[a-z0-9.-]{0,64}")
_SLUG_FIELDS: Final[tuple[str, ...]] = ("query_source",)
# The timestamptz columns: clamped to the epoch sentinel rather than to a bound,
# since a value outside the range is a corrupt clock, not a large one.
_EPOCH_FIELDS: Final[tuple[str, ...]] = ("played_at", "queued_at")
_INT4_MAX: Final[int] = 2**31 - 1
_INT8_MAX: Final[int] = 2**63 - 1
# 9999-12-31T23:59:59Z. Public because it is the epoch domain of the play_history
# timestamptz columns, not just this validator's: history_archive clamps cutoffs to
# it, and migrations/0001_play_history.sql spells the same value into its CHECKs.
TS_MAX: Final[float] = 253402300799.0


@dataclass(frozen=True, slots=True, kw_only=True)
class HistoryEntry:
    """One played song at rest — an element of guild:{id}:history.

    Zero-values mean "unknown": absent wire fields default on parse and the display
    layer degrades. The field set matches the Postgres play_history row.

    guild_id is redundant on the per-guild display list (the key carries it) but
    required on the global history:outbox stream, where all guilds interleave and
    the drainer maps each entry to a Postgres row. Entries written before the field
    existed parse as guild_id=0.

    message_id and channel_id are a WEAK reference, taken off the same message at
    song end so the pair is both real or both 0: the NP host migrates across
    messages and channels during one song and a dedicated host is deleted when
    retired, so it is neither a foreign key nor part of play_history_dedup.
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
    # per fragment: an interjection's resume tail inherits the interrupted stamp,
    # so an interrupted play files under when the listener first heard it.
    played_at: float = 0.0
    message_id: int = 0  # NP host at song end; 0 = unknown (see class docstring)
    channel_id: int = 0  # the channel that host was in; 0 = unknown, always paired
    queued_at: float = 0.0  # unix epoch when the user ASKED; 0 = unknown
    # Songs ahead of it at ask time, counting the one playing. 0 means it
    # played immediately — and is also what a pre-feature entry parses as.
    queue_position: int = 0
    # How the song was asked for: "search", or the host of the link that was
    # pasted. "" = unknown, which is what every pre-feature entry parses as.
    # Classified at parse time (src.sources) because it is not recoverable from
    # webpage_url — a Spotify link and a plaintext search both resolve to a
    # YouTube watch URL.
    query_source: str = ""

    def __post_init__(self) -> None:
        """Normalize into the play_history column domain at construction.

        The schema lock: an instance is, by construction, a row Postgres accepts.
        Every producer routes through here — from_song, parse_history_entry,
        _row_to_entry, dataclasses.replace, the backfill — so no consumer re-proves
        it and the archive holds no opinion about data.

        One field is exempt: the clamp floors every integer at 0, but play_history's
        CHECK on guild_id is strictly `> 0`, so a guild_id-0 entry is constructible
        and not insertable. Clamping up to 1 would file an unattributable play into
        a real guild's history; relaxing the CHECK would make guild 0 a permanent
        bucket of orphans no read path excludes. Refusing at the database routes
        those entries to play_history_rejected, where a row means "a producer
        stopped stamping guild_id" — the actual defect.

        These are the only ways a wire-parseable entry could fail an INSERT:
        everything else the columns could refuse — lone surrogates, non-finite
        floats, integers past 64 bits — orjson already refuses to encode. Range
        tests rather than pairs of bound checks because a chained comparison is
        False for NaN, which lands NaN on the sentinel.

        Total BY DESIGN — never raises, and it runs on the read path over stored
        rows that predate it, so a validator that rejected them would break -history
        for precisely the guilds with the most history; strictness lives in the DB
        CHECK constraints instead. No type coercion: HistoryEntry(title=None)
        raising TypeError out of the `in` test below is correct — coercing would
        store a row reading "None".
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
        """Canonical extraction from a finished song. guild_id, message_id and
        channel_id are keyword-required because the song doesn't carry them and a
        forgotten stamp writes cleanly as 0 (play_history's CHECKs are `>= 0`),
        permanently indistinguishable from a song that genuinely had no host — pass
        0 explicitly for that, and take both ids off the same message.

        played_at rides the song rather than a caller clock: the loop stamps it at
        vc.play() and every later fragment inherits it, so an interrupted song files
        once, under the moment it actually started.

        played_secs is the position reached (start_offset + audio delivered), capped
        at duration when known. An interrupted song is recorded once at its
        resume tail, whose position spans the full listened range; a ?t= song
        includes the skip.
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
            played_at=song.played_at,
            message_id=message_id,
            channel_id=channel_id,
            queued_at=song.analytics.queued_at,
            queue_position=song.analytics.queue_position,
            query_source=song.query_source,
        )

    @classmethod
    def from_queue_object(cls, item: QueueObject, *, guild_id: int) -> Self:
        """A played song recorded as it LEAVES the queue, rather than as it ends.

        The -clear/-remove counterpart to from_song. There is no YTDL to hand it:
        this entry was interrupted by an interjection and destroyed before its tail could
        play, so the queue object is all that is left of it.

        played_secs comes from `ts`, the resume offset, which is ABSOLUTE (see
        YTDL.position_secs = start_offset + elapsed) — it already spans everything
        heard across every earlier fragment. Capped at duration like from_song's.

        The host ids come off the tail's np_* fields, which name the card its
        interrupted fragment left frozen. Still resolvable here: the cleanup that
        deletes that card fires only when a tail STARTS, and a flushed tail never
        does. 0/0 when the tail was never stamped.
        """
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
        """Serialize to the wire format. Field table spelled out so the wire schema
        is pinned to HistoryEntryField, not to Python attribute names."""
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
    """Deserialize one history-list entry. Corrupt entries (bad JSON, wrong type,
    malformed fields) are dropped with a warning, as in parse_queue_entry. Unknown
    keys are ignored and missing keys default, so mixed-build readers stay tolerant
    in both directions."""
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
    """A guild's complete persisted playback aggregate — state hash, pending queue,
    now-playing snapshot, played-song history — read together in one pipelined
    round-trip (GuildRedisStore.get_playback_snapshot).

    "A guild owns a queue (and a history)" as a type: recovery decisions spanning
    the halves live here as named properties, not expressions at call sites.
    """

    state: GuildStateData
    queue: tuple[QueueEntry, ...] = ()
    # None when no song was playing (the hash is DELETE'd wholesale on song
    # end, so empty == no song — same contract as NowPlayingData.from_redis).
    now_playing: NowPlayingData | None = None
    # Newest-first, as stored (GuildHistory.restore() handles the reversal).
    history: tuple[HistoryEntry, ...] = ()
    # The guild's durable settings, read in the same round trip because restore is
    # exactly when they are needed.
    config: GuildConfig = GuildConfig()

    @property
    def stored_volume(self) -> float | None:
        """The guild's volume, or None if it never set one.

        Config first, then the legacy state field. The fallback is a one-release
        migration path: volume used to live in guild:{id}:state, and dropping it
        outright would silently reset every deployed guild to 100%. Restore SEEDS
        config from whatever it finds here (migrate_volume, HSETNX — never an
        overwrite), and set_volume keeps both copies fresh, so the two agree in
        both directions and a rollback is a no-op. Delete this leg, StateField.VOLUME
        and set_volume's legacy write together once every deployment has started once.
        """
        if self.config.volume is not None:
            return self.config.volume
        return self.state.volume

    @property
    def pending_count(self) -> int:
        return len(self.queue)

    @property
    def has_restorable_playback(self) -> bool:
        """The restore_guild() gate: True when a restart has anything to resume
        — pending queue entries or a song that was mid-play at crash time."""
        return bool(self.queue) or self.state.has_crashed_song


@dataclass(frozen=True, slots=True, kw_only=True)
class GuildRecoveryGate:
    """The minimal read `restore_guild()` needs to decide whether to reconnect: the
    state hash plus the pending queue's *length* — never its contents.

    _restore_state re-reads the full GuildPlaybackSnapshot after a successful voice
    connect. Reading only the length keeps a -stopped guild's (possibly long)
    leftover queue off the wire on every on_ready.
    """

    state: GuildStateData
    pending_count: int = 0

    @property
    def has_restorable_playback(self) -> bool:
        """True when a restart has anything to resume — pending entries or a song
        mid-play at crash time. GuildPlaybackSnapshot.has_restorable_playback over
        the queue length."""
        return self.pending_count > 0 or self.state.has_crashed_song


# ── -analytics: the render worker's boundary payload ──────────────────────────
# Unpickling a dataclass imports its defining module, so their home decides what a
# chart worker drags in — this module is stdlib + orjson.
# See docs/ARCHITECTURE.md#analytics-rendering.
#
# Every field defaults, so an entry written by an older build still decodes through
# analytics_card.from_cache.


@dataclass(frozen=True, slots=True, kw_only=True)
class DailyPoint:
    """One x-position of the per-day series. `day` is an ISO date (`YYYY-MM-DD`)
    naming the bucket's first UTC day — at --days 365 the bucket is a week, and
    the date is its Monday."""

    day: str = ""
    plays: int = 0
    listen_secs: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceDay:
    """One segment of a stacked bar: how many plays a query_source contributed to
    one bucket. Sparse — a source with no plays that day has no row."""

    day: str = ""
    source: str = ""
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class HeatCell:
    """One weekday x hour-of-day cell. `dow` is ISO (1 = Monday), `hour` is 0-23,
    both in UTC — the frame play_history was written in."""

    dow: int = 0
    hour: int = 0
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class CompletionBucket:
    """Plays whose played/duration ratio fell in one tenth, per source. `bucket` is
    1-10 covering [0,0.1) .. [0.9,1.0]; the SQL folds width_bucket's overflow 11
    into 10 because a ratio of exactly 1.0 is the modal value."""

    source: str = ""
    bucket: int = 0
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class DurationBucket:
    """Plays by song length in whole minutes, 0-19, with 20 meaning "20 min or
    longer" — an open top bucket, since an hour-long mix must not stretch the axis
    over every three-minute song."""

    minutes: int = 0
    plays: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class SourceCompletion:
    """DURATION-WEIGHTED completion for one source: the sums, not the ratio, so the
    renderer divides once and the zero case is visible rather than encoded.

    This is a different question from CompletionBucket and the two disagree by
    design — on the live archive pasted YouTube links play 21% of their seconds but
    14 of 16 individual plays run to the end, because a handful of abandoned
    hour-long mixes dominate the weighted number.
    """

    source: str = ""
    played_secs: int = 0
    duration_secs: int = 0


@dataclass(frozen=True, slots=True, kw_only=True)
class TopListener:
    """One row of the embed's listeners section. Distinct from history_archive's
    RequesterLeader, which serves -leaderboard: this one may not live in a module
    the render worker would have to import (see the block comment above)."""

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


# Sentinel for "no usable queue-wait data in this window". Negative because 0 is a
# legitimate wait — a song queued into an empty queue — and the two must not collapse.
WAIT_UNAVAILABLE: Final[float] = -1.0
# The queue-wait percentiles and the median's index within them. The SQL array, the
# length checks and the figure's labels all read from here.
WAIT_PERCENTILES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5, 0.75, 0.9)
WAIT_MEDIAN_INDEX: Final[int] = WAIT_PERCENTILES.index(0.5)


@dataclass(frozen=True, slots=True, kw_only=True)
class AnalyticsMetrics:
    """Everything -analytics knows about one guild and one window.

    The whole object crosses to the chart worker, including the three top-N lists
    it never draws: they are ~2 KB against a ~15 KB payload, and one boundary type
    is worth more than the split. What the worker may NOT draw is any of their
    strings — the runtime image has no CJK, Thai or emoji glyphs, so a human-authored
    name renders as tofu, silently. _ascii_safe() enforces that in analytics_render.

    `days` is the REQUESTED window and archived_days its real coverage; the title
    names both when they differ, because a 30-day frame that is 93% empty is a worse
    answer than a 2-day frame that says so.
    """

    days: int = 0
    # Both epochs, so the footer and the cache TTL read the same clock the buckets
    # did. window_end is exclusive and is the day (or, at --days 365, the week)
    # boundary the window stops at — never "now".
    window_start_epoch: float = 0.0
    window_end_epoch: float = 0.0
    # "day" or "week". At 365 days a daily series would be 365 sub-pixel bars, so
    # the SQL downsamples; the y-axis label has to move with it.
    bucket_unit: str = "day"
    # Today's UTC midnight, from the same clock read as the bounds. The cache expiry
    # derives from this, so it cannot outlive the day the aggregate covers.
    today_start_epoch: float = 0.0
    archived_days: int = 0

    plays: int = 0
    listen_secs: int = 0
    unique_songs: int = 0
    unique_listeners: int = 0
    unique_artists: int = 0
    wait_p50_secs: float = WAIT_UNAVAILABLE
    # Plays with duration_secs <= 0 — livestreams. Counted everywhere else and
    # excluded from the completion panel alone, which names the number so a guild
    # with many of them is not silently reading a partial chart.
    livestream_plays: int = 0

    daily: tuple[DailyPoint, ...] = ()
    daily_by_source: tuple[SourceDay, ...] = ()
    heat: tuple[HeatCell, ...] = ()
    completion: tuple[CompletionBucket, ...] = ()
    durations: tuple[DurationBucket, ...] = ()
    source_completion: tuple[SourceCompletion, ...] = ()
    # p10/p25/p50/p75/p90 of queue wait, seconds. Empty when no row in the window
    # had a non-sentinel queued_at.
    wait_pcts: tuple[float, ...] = ()

    top_listeners: tuple[TopListener, ...] = ()
    top_artists: tuple[TopArtist, ...] = ()
    top_songs: tuple[TopSong, ...] = ()

    @property
    def is_empty(self) -> bool:
        """Whether the window holds nothing. Reads the explicit play count, never a
        branch's shape: every aggregate coalesces to `[]`, so an empty daily series
        also describes a guild whose rows all failed one branch's filter."""
        return self.plays == 0

    @property
    def period_label(self) -> str:
        """The window as a period, in the unit it is actually bucketed by.

        365 mod 7 is 1, so `date_trunc('week', end - 365 days)` snaps back six days
        and the weekly window covers 53 whole weeks — 371 days. Calling that "last
        365 days" names six fewer days than the chart draws. Lives here because the
        card and the figure must agree and the figure cannot import the card."""
        if self.bucket_unit == "week":
            weeks = -(-self.days // 7)
            return f"last {weeks} week{'s' if weeks != 1 else ''}"
        return f"last {self.days} day{'s' if self.days != 1 else ''}"
