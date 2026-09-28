"""
The queue item — one live entry of a guild's queue, resolved or not, and the two
values it carries that are not part of the ask itself: the analytics stamped at ask
time and the Now Playing card an interrupted fragment leaves for its resume tail.
Pure data: GuildQueue owns the items, guild_state.py owns their bytes at rest, and
youtube.py builds them from what yt-dlp finds.
"""

from dataclasses import dataclass, field
from typing import Final, Optional, Union

import discord


# ── Pure-analytics values, grouped ───────────────────────────────────────────


@dataclass(frozen=True, slots=True, kw_only=True)
class Analytics:
    """Values carried on live queue objects (QueueObject, YTDL) for
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


@dataclass(frozen=True, slots=True, kw_only=True)
class NpHostRef:
    """The live Now Playing host an interrupted fragment left behind, for its
    resume tail to dispose of. Runtime only: a Message cannot be serialized and
    own_embeds cannot be rebuilt from ids, so the wire fields alone can never
    strip-edit a retirement (MusicPlayer._retire_np_host)."""

    message: discord.Message
    own_embeds: list[discord.Embed]
    dedicated: bool


# slots: a 10,000-track Spotify playlist holds one of these per track while its
# searches wait to resolve — 216 B each by sys.getsizeof on this interpreter,
# against 344 B for the same instance carrying a __dict__. Keep the class off
# asdict (it deep-copies requester), vars (it raises) and any pickle path.
# eq=False: an item is one ask, so it compares and hashes by identity — two asks
# for one song are two items in a set as on the deque, and GuildQueue keys its
# swap records on the object itself.
@dataclass(frozen=True, slots=True, kw_only=True, eq=False)
class QueueObject:
    """One queued song, resolved or not. A track queued from a Spotify playlist
    arrives as a search: `search` set, `webpage_url` its own Spotify page or empty.
    The resolve at dequeue returns it with what yt-dlp found over its display fields
    and `search` cleared, leaving the queued original on the deque. Everything else
    about the ask is the same either way, which is why there is one type — see
    docs/ARCHITECTURE.md#one-queue-item.

    Every change to a queued item is a `replace()` copy: the queue swaps it into the
    item's slot (`GuildQueue.replace_item`) and a playing song holds its own
    (`YTDL.queued`)."""

    webpage_url: str
    title: str
    requester: Union[discord.User, discord.Member]
    ts: Optional[int] = None
    user_input: Optional[str] = None
    duration: Optional[int] = None  # seconds, from yt-dlp at enqueue time
    uploader: Optional[str] = None  # YouTube channel name
    thumbnail: Optional[str] = None
    # False only for the crash-recovered head restore_crashed() re-queues: it was
    # never RPUSHed to the Redis list, so the loop must skip its redis_pop_for().
    # Read via guild_queue.is_persisted().
    persisted: bool = True
    # ── interjection flags ──
    # `interjected` is attribution only (span attribute); `is_resume` marks the
    # rebuilt tail of an interrupted song (ts = interrupt position) and selects
    # the "Resuming…" notice; `start_paused` re-pauses right after vc.play() so
    # a song paused at interjection returns parked.
    interjected: bool = False
    is_resume: bool = False
    start_paused: bool = False
    # A -replay copy. Runtime-only, so absent from SongQueueEntry: a crash restores
    # an ordinary queued song. YTDL carries it so _neutralize_prefetch's rebuild
    # keeps it; an interjection's resume tail never sets it.
    is_replay: bool = field(default=False, repr=False)
    # Ask-time analytics. yt_source/yt_playlist REQUIRE it; the default exists
    # for rehydration and the carry sites, which always pass a real value.
    analytics: Analytics = ANALYTICS_ZERO
    # "search", or the host of the pasted link; "" = unknown (src.sources).
    query_source: str = ""
    # Epoch when the audio started, stamped by the loop at vc.play(). A resume
    # tail INHERITS it so every fragment of one play records the same start;
    # 0.0 = not played yet.
    played_at: float = 0.0
    # Stream-retry state, runtime only (never on the Redis wire): plays already
    # spent on this song whose stream never opened, and the formats that failed.
    # A crash resets both. See MusicPlayer._retry_failed_stream.
    stream_attempts: int = 0
    failed_format_ids: frozenset[str] = frozenset()
    # The NP card the interrupted fragment left frozen, set on a resume tail at
    # the fragment's iteration end and consumed when the tail starts. The ids
    # survive a restart, the ref does not, and only the ref can strip-edit a
    # response host. 0/0/False = nothing to clean up.
    np_message_id: int = 0
    np_channel_id: int = 0  # from message.channel.id — NEVER the home channel
    np_dedicated: bool = False  # a pure NP message (deletable) vs a response
    np_host_ref: Optional[NpHostRef] = field(default=None, repr=False)
    # The `ytsearch:` term an unresolved item still has to resolve, cleared by the
    # resolve at dequeue. `title` meanwhile is the walk's row name, or empty when
    # the walk had none — every renderer falls back to this term.
    search: str = ""

    @property
    def unresolved(self) -> bool:
        """True while this item is a search: nothing may stream it, and its Redis
        entry carries the term under `ytsearch`."""
        return bool(self.search)
