"""
The queue item — one live entry of a guild's queue, resolved or not, with the
analytics stamped on it at ask time and the Now Playing card an interrupted
fragment leaves for its resume tail.
Pure data: GuildQueue owns the items, guild_state.py owns their bytes at rest, and
youtube.py builds them from what yt-dlp finds.
"""

from dataclasses import dataclass, field
from typing import Optional, Union

import discord


@dataclass(frozen=True, slots=True, kw_only=True)
class NpHostRef:
    """The live Now Playing host an interrupted fragment left behind, and the
    embeds that are its own. Runtime only: a Message cannot be serialized and
    own_embeds cannot be rebuilt from ids, so the card's ids alone can never
    strip-edit a retirement (MusicPlayer._retire_np_host)."""

    message: discord.Message
    own_embeds: list[discord.Embed]


@dataclass(frozen=True, slots=True, kw_only=True)
class NpCard:
    """The Now Playing card an interrupted fragment left frozen, for its resume
    tail to dispose of when it starts. The ids survive a restart and can delete
    a DEDICATED card (a pure NP message, as against a command response); the
    live ref is what allows a strip-edit, and does not survive one."""

    message_id: int
    channel_id: int  # from message.channel.id — NEVER the home channel
    dedicated: bool
    host_ref: Optional[NpHostRef] = field(default=None, repr=False)


# slots: a 10,000-track Spotify playlist holds one of these per track while its
# searches wait to resolve — 200 B each by sys.getsizeof on this interpreter,
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
    # A -restart copy. Runtime-only, so absent from SongQueueEntry: a crash restores
    # an ordinary queued song. YTDL carries it so _neutralize_prefetch's rebuild
    # keeps it; an interjection's resume tail never sets it.
    is_restart: bool = field(default=False, repr=False)
    # ── ask-time analytics ──
    # Stored and carried, never branched on or rendered; the wire entry and the
    # play_history row hold both under these names. yt_source/yt_playlist
    # REQUIRE them; the defaults are what a pre-feature wire entry rehydrates as.
    # Unix epoch when the user ASKED: the command message's snowflake time
    # (Discord's clock, so played_at - queued_at can go slightly negative).
    queued_at: float = 0.0
    # Songs ahead at ask time, counting the one playing (0 = played immediately).
    # Read once at dispatch, so it is approximate against the insert.
    queue_position: int = 0
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
    # The card the interrupted fragment left frozen, set on a resume tail at the
    # fragment's iteration end and consumed when the tail starts. None = nothing
    # to clean up.
    np_card: Optional[NpCard] = None
    # The `ytsearch:` term an unresolved item still has to resolve, cleared by the
    # resolve at dequeue. `title` meanwhile is the walk's row name, or empty when
    # the walk had none — every renderer falls back to this term.
    search: str = ""
    # The recording this item names, when the walk knew one. Read only by the
    # resolve, which searches it before the term (youtube._search_terms) and
    # clears it with the term.
    isrc: Optional[str] = None

    @property
    def unresolved(self) -> bool:
        """True while this item is a search: nothing may stream it, and its Redis
        entry carries the term under `ytsearch`."""
        return bool(self.search)
