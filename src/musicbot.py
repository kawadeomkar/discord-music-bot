import asyncio
import contextlib
import re
import time
from dataclasses import dataclass, replace
from enum import Enum
from itertools import islice
from typing import (
    Any,
    Final,
    Optional,
    Union,
    assert_never,
    cast,
)
from collections.abc import AsyncGenerator, Awaitable, Callable, Coroutine, Sequence

import discord
from discord.ext import commands

import redis.asyncio as aioredis

from src.config import (
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
    debug_prometheus_url,
    history_archive_enabled,
    spotify_enabled,
    using_default_postgres_password,
)
from src import debug as debug_mode
from src import leaderboard
from src.guild_history import history_embeds
from src.guild_state import Analytics
from src.leaderboard import LeaderboardFlags
from src.history_archive import (
    ArchiveReader,
)
from src.musicplayer import MusicPlayer
from src.redis_client import (
    HISTORY_CACHE_LIMIT,
    GuildRedisStore,
    cache_get,
    cache_set,
)
from src.guild_queue import QueueItem, RemoveMode, RemoveOutcome, item_label
from src.sources import (
    QUERY_SOURCE_SEARCH,
    timestamp_warning,
    unquote_argument,
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    query_source_of,
    spotify_playlist_to_ytsearch,
)
from src.spotify import (
    Spotify,
    SpotifyAuthError,
    SpotifyRateLimitError,
    SpotifyRequestError,
)
from src.youtube import YTDL, ExtractionError, QueueObject
from contextvars import Token

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

from src.ping import run_health_dashboard, send_latency_line
from src.recovery import VoiceWatchdog, restore_guild
from src.telemetry import get_tracer
from src.util import (
    EMBED_FIELD_LIMIT,
    background_typing,
    cancel_task,
    fmt_duration,
    notice_embed,
    pluralize,
    queue_message,
    safe_label,
    record_span_error,
    send_embed,
    spawn_background,
    trace_footer,
    truncate,
    truncate_embed_title,
    get_logger,
)

log = get_logger(__name__)
_tracer = get_tracer(__name__)


class SpotifyDisabledError(Exception):
    """Raised when a Spotify link is played but Spotify support isn't usable.
    Carries the SpotifyStatus so the message can separate no credentials configured
    (disabled) from configured but rejected at startup (invalid). The message is
    user-facing — _command_error renders it into the error embed."""

    def __init__(self, status: SpotifyStatus) -> None:
        self.status = status
        if status is SpotifyStatus.INVALID:
            message = (
                "Spotify links aren't available right now — this bot has Spotify "
                "credentials configured, but Spotify rejected them at startup "
                "(check SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET). "
                "Try a YouTube or SoundCloud link, or just search by name."
            )
        else:  # disabled (and any unexpected value — safest generic message)
            message = (
                "Spotify links aren't available on this bot — it was started without "
                "Spotify credentials (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET). "
                "Try a YouTube or SoundCloud link, or just search by name."
            )
        super().__init__(message)


class PlaylistInputError(ValueError):
    """A playlist link the user can fix by editing it, rather than a bot failure.

    Subclasses ValueError so nothing that already catches one changes behaviour,
    and carries user_message so _command_error renders actionable copy instead of
    a "ValueError: …" line. One base class so _command_error's tuple names the
    concept rather than growing an entry per case.
    """

    def __init__(self, log_message: str, user_message: str) -> None:
        super().__init__(log_message)
        self.user_message = user_message


class PlaylistIndexError(PlaylistInputError):
    """`index=` names a position past the end of the playlist. Both numbers are
    in the message: the user cannot fix the link without knowing the real one."""

    def __init__(self, index: int, total: int) -> None:
        self.index = index
        self.total = total
        # A one-song playlist has no range to offer — "from 1 to 1" reads as a
        # bug in the message rather than advice.
        fix = (
            "Drop the `&index=` from the link to queue it."
            if total == 1
            else (
                f"Pick a position from 1 to {total}, or drop the `&index=` "
                f"from the link to queue the whole playlist."
            )
        )
        super().__init__(
            f"playlist index {index} past end ({total} tracks)",
            f"That link starts the playlist at **#{index}**, but the playlist "
            f"only has **{total} {pluralize(total, 'song')}** — nothing was "
            f"queued.\n\n{fix}",
        )


class EmptyPlaylistError(PlaylistInputError):
    """A playlist that resolved to nothing queueable. Deliberately vague about
    which cause: yt-dlp drops unavailable entries before this code sees them, so
    "empty" and "every video is private" are indistinguishable here."""

    def __init__(self) -> None:
        super().__init__(
            "playlist resolved to no tracks",
            "That playlist has no songs I can queue — it may be empty, or every "
            "video in it may be private or unavailable.",
        )


HISTORY_MIN_LIMIT = 1
# Pinned to HISTORY_CACHE_LIMIT. recent() serves this command from the Redis list
# alone, which holds exactly that many entries, so a larger ceiling here returns a
# short page instead of failing. Raise both together or neither.
HISTORY_MAX_LIMIT = HISTORY_CACHE_LIMIT
# 8 song embeds + the ≤2-embed NP block MusicContext.send may prepend = Discord's
# per-message cap of 10, so the block always fits and is never shed.
HISTORY_EMBEDS_PER_MESSAGE = 8

# How long a cold-start command (-play, -resume) waits for its restore. Generous for
# one pipelined read; bounded because the pool sets no socket_timeout, so a server
# that accepts the connection then stalls would hang the command outright.
RESTORE_WAIT_SECS = 5.0

# How long a warm -play waits for its restore before reading the ask-time queue
# depth. Short because a timeout here only costs an approximate analytics field:
# the depth is recorded from whatever has landed and the command proceeds.
DEPTH_RESTORE_WAIT_SECS = 1.0


class HistoryFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    limit: int = 10


NOW_FLAG: Final[str] = "--now"
NEXT_FLAG: Final[str] = "--next"


class PlayMode(Enum):
    """Where a `-play` invocation puts its song.

    One field rather than a bool per flag: the flags contradict each other, and two
    bools would make `now and next` representable — a state every branch downstream
    would then have to decide about.
    """

    NORMAL = "normal"
    NOW = "now"
    NEXT = "next"


_FLAG_MODES: Final[dict[str, PlayMode]] = {
    NOW_FLAG: PlayMode.NOW,
    NEXT_FLAG: PlayMode.NEXT,
}


class Placement(Enum):
    """Where an enqueue puts its songs, AND which confirmation says so.

    One value rather than a `front: bool`, because those are two decisions that
    only look like one. `build_resume_notice_embed` — "❗ Resumed from queue …
    N songs from the previous session resume after it" — is true for a
    disconnected bot waking a persisted queue and false in every clause for a warm
    front-insert, and it returns None only when the queue is EMPTY, so it renders
    exactly on the case that would be wrong.
    """

    TAIL = "tail"
    COLD_FRONT = "cold_front"
    NEXT = "next"


# Every dash Unicode offers that a keyboard or a paste substitutes for ASCII `-`:
# hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar. iOS
# turns a typed `--` into a single em dash on its own, which is the common way a
# correctly-typed flag still arrives spelled wrong.
_DASHES: Final[str] = "-‐‑‒–—―"
# Alternation built from _FLAG_MODES' own keys, so a renamed flag cannot leave this
# branch offering one that no longer exists. The group is a flag minus its dashes;
# split_play_args re-attaches them for the reply.
_NEAR_FLAG_RE: Final[re.Pattern[str]] = re.compile(
    f"[{_DASHES}]{{1,2}}({'|'.join(flag[2:] for flag in _FLAG_MODES)})"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayArgs:
    """`-play`'s argument, split into the placement flag and the query behind it.

    kw_only because `query` and `dash_typo` are adjacent, both `str`-ish, and one of
    them is echoed straight into an embed: positionally they transpose silently, and
    the result is the bot asking "did you mean `<the user's whole search>`?".

    `dash_typo` is the third state: the leading token is recognisably a misspelt flag
    rather than either a flag or a search, so the command asks instead of guessing. It
    names the flag meant, and only ever accompanies `PlayMode.NORMAL` — the whole
    point is that we do not know which the user meant.
    """

    mode: PlayMode
    query: str
    dash_typo: Optional[str] = None


def split_play_args(argument: str) -> PlayArgs:
    """Split a leading `--now`/`--next` off `-play`'s argument.

    Only the FIRST token is considered, so a flag further along stays part of the
    search text. Stripping it from anywhere would silently eat the token out of a
    legitimate search, and would leave the origin `-remove` matches on as the line
    minus a hole rather than as something the user typed. One flag, never a run:
    `-p --now --next x` takes `--now` and searches for "--next x", exactly as a
    second `--now` would.

    Hand-parsed rather than a FlagConverter (as in src/debug.py's parse_debug_arg, and
    unlike HistoryFlags above): the converter's grammar is `--flag value`, which cannot
    express a valueless switch, and it matches flags anywhere in the line — which is
    exactly the leading-token rule this has to enforce.

    A leading token that is only a dash away from a flag gets `dash_typo`, because
    `-now` or an autocorrected `—next` cannot be anything else. The exact-match lookup
    runs first for that reason: a real `--now` also satisfies the near-miss pattern. A
    bare leading `now`/`next` deliberately does NOT: `-p now thats what i call music`
    and `-p next to me` are real searches, and guessing there would break them.
    """
    stripped = argument.strip()
    parts = stripped.split(maxsplit=1)
    if not parts:
        return PlayArgs(mode=PlayMode.NORMAL, query="")
    head = parts[0].lower()
    # No strip on the tail: `stripped` had none, and split() eats the separator run.
    rest = parts[1] if len(parts) > 1 else ""
    mode = _FLAG_MODES.get(head)
    if mode is not None:
        return PlayArgs(mode=mode, query=rest)
    typo = _NEAR_FLAG_RE.fullmatch(head)
    if typo is not None:
        return PlayArgs(
            mode=PlayMode.NORMAL, query=stripped, dash_typo=f"--{typo.group(1)}"
        )
    return PlayArgs(mode=PlayMode.NORMAL, query=stripped)


# Bound on one echoed needle, which owns a field to itself. Discord renders
# markdown in field values, so what a user typed goes through safe_label first.
_ECHO_MAX = 200

# One row of a multi-row field — ten of these share the budget one needle gets.
_ECHO_ROW_MAX = 70


# The most dropped positions worth spelling out; past this the list says nothing
# the count above it did not.
_MAX_SHOWN_POSITIONS = 60


def _echo(text: str, limit: int = _ECHO_MAX) -> str:
    """A needle safe to put in an embed — see util.safe_label."""
    return safe_label(text, limit)


def _field(value: str) -> str:
    """An embed field value that cannot 400 the send. The callers below build from
    lists whose length is the user's to choose, and the send happens AFTER the
    queue has been mutated."""
    return truncate(value, EMBED_FIELD_LIMIT)


def _matched_label(outcome: RemoveOutcome, needle: str) -> str:
    """How the removal matched, for the reply's "Matched" field. An origin match
    names which of the user's own inputs did it, since one argument can take out a
    whole playlist."""
    # Not wrapped in a code span: inside one Discord renders safe_label's
    # backslashes literally, so `-remove foo_bar` comes back as `foo\_bar`.
    shown = _echo(needle)
    if outcome.mode is not RemoveMode.ORIGIN:
        return shown
    kinds = {item.query_source for item in outcome.removed if item.query_source}
    # Only when every removed item agrees — a mixed set has no one kind to name.
    kind = kinds.pop() if len(kinds) == 1 else ""
    them = "them" if len(outcome.removed) > 1 else "it"
    if kind == QUERY_SOURCE_SEARCH:
        return f"{shown} — the search you queued {them} with"
    return f"{shown} — the {kind + ' ' if kind else ''}link you queued {them} with"


@dataclass(frozen=True, slots=True, kw_only=True)
class ActiveCommand:
    """The bookkeeping cog_before_invoke opens and cog_after_invoke closes.

    `token` is what otel_context.attach() returned and detach() requires back
    (`object` does not satisfy it). `started` is monotonic and exists for the debug
    footer, which reports elapsed time AT EACH SEND — so a command that sends twice
    shows two increasing numbers, timing its phases.
    """

    span: Span
    token: Token[Context]
    started: float


@dataclass
class ResolvedSpotifyPlaylist:
    """A Spotify playlist resolved to track titles — still needs per-title
    YouTube search resolution before it can be queued."""

    titles: list[str]


@dataclass
class ResolvedYoutubePlaylist:
    """A YouTube playlist already resolved to playable QueueObjects.

    `skipped` is how many leading tracks the URL's `index=` dropped, carried for
    the enqueue embed alone: tracks is already sliced, so nothing downstream has
    to know the playlist started anywhere but its own first entry.
    """

    tracks: list[QueueObject]
    skipped: int = 0


def _apply_playlist_index(
    tracks: list[QueueObject],
    index: Optional[int],
) -> tuple[list[QueueObject], int]:
    """Drop the tracks ahead of YouTube's 1-based `index=`, returning what is left
    and how many went. A share link copied mid-playlist carries the position it was
    copied at, so playing it starts there rather than back at track 1.

    An index past the end raises PlaylistIndexError rather than queueing nothing:
    the user named a position this playlist does not have, and an empty enqueue
    reports success. The empty-playlist guard lives here too, so both callers
    get it.
    """
    if not tracks:
        raise EmptyPlaylistError
    if index is None or index <= 1:
        return tracks, 0
    if index > len(tracks):
        raise PlaylistIndexError(index, len(tracks))
    kept = tracks[index - 1 :]
    dropped = index - 1
    # Positions were assigned at construction, before this slice. Rebase to
    # kept-relative: the dropped tracks never enqueue, so an &index=N link would
    # otherwise record every kept track N-1 too deep.
    for track in kept:
        track.analytics = replace(
            track.analytics,
            queue_position=track.analytics.queue_position - dropped,
        )
    return kept, dropped


def _apply_playlist_timestamp(tracks: list[QueueObject], source: YTSource) -> None:
    """Start the first queued track at the link's `t=` offset — but only when
    that track is the video the link actually names.

    A playlist link carries one offset and N tracks, so the offset belongs to the
    `v=` video alone. Without a matching `index=` the queue starts at track 1,
    which is usually a different song, and seeking that one would be wrong. The
    offset was previously parsed and then dropped on this path.
    """
    if not source.ts or not source.video_id or not tracks:
        return
    # Substring, not equality: yt_playlist takes the entry's own `url` when it
    # has one, so the shape it built is not guaranteed. An 11-char video id
    # matching some other part of a YouTube URL is not a case that arises.
    if source.video_id in tracks[0].webpage_url:
        tracks[0].ts = source.ts


def _plays_after_note(
    mp: MusicPlayer, voice_client: Optional[discord.VoiceProtocol]
) -> str:
    """What a `--next` confirmation says about when the song will be heard.

    Names the song it waits behind rather than quoting a time — see the comment at
    the call site for why an ETA is the wrong tool for the very next slot. The
    paused sentence is not decoration: `--next` deliberately does NOT interject a
    paused song (the request is not buried behind it, it IS next), so the bot stays
    silent afterwards and nothing else would say why.
    """
    current = mp.current_song
    if current is None:
        # A claim with no current_song is loop() between taking the prefetch result
        # and starting it — a song IS about to play, and put_front inserts behind
        # it, so "starts now" would be wrong twice over.
        if mp.queue.claim_outstanding():
            return "Plays after the song starting now."
        return "Nothing is playing, so it starts now."
    note = f"Plays after **{current.title or 'the current song'}**."
    if isinstance(voice_client, discord.VoiceClient) and voice_client.is_paused():
        note += " Playback is paused — `-resume` to carry on."
    return note


def _with_queue_position(item: QueueItem, position: int) -> QueueItem:
    """Re-mint one item's ask-time `queue_position`.

    QueueObject is mutable and YTSource is frozen, so one is stamped in place and
    the other returns a copy — the caller must use the return value either way.
    """
    analytics = replace(item.analytics, queue_position=position)
    if isinstance(item, QueueObject):
        item.analytics = analytics
        return item
    return replace(item, analytics=analytics)


def _collection_note(
    url: str, queued: int, *, returns: str = "", head_playing: bool
) -> str:
    """What a `-play` that queued a whole collection owes the user: how many tracks
    landed, when the interrupted song comes back, and the one-command undo.

    One builder for the three paths that queue a collection — the interjection, the
    one whose song ended mid-resolve, and the one resumed mid-resolve — because they
    make the same promise about the same tracks, and only one of them used to say
    anything at all. The other two queued a whole playlist behind a reply that named
    a single song.

    `head_playing` is not cosmetic. A playing song has no queue object — its entry
    was LPOPed when it started — so `-remove` cannot reach it and only `-skip` will.
    Offering "takes the whole playlist back out" there leaves behind the one track
    the user is most likely trying to undo.
    """
    undo = (
        "the queued ones back out; the one playing needs `-skip`."
        if head_playing
        else "the whole playlist back out."
    )
    return (
        f"\n\nQueued **{queued}** {pluralize(queued, 'song')} from the playlist."
        f"{returns}\nNot what you wanted? `-remove {_echo(url)}` takes {undo}"
    )


def _front_insert_depth(mp: MusicPlayer) -> int:
    """Ask-time `queue_position` for a song going to the FRONT of the queue: it
    waits behind the song playing and nothing else.

    One definition for the two callers — `-play --next`, and the interjection whose
    song ended before it could be interrupted — because they front-insert into the
    same queue and a drift between them would write two different numbers to
    Postgres for the same situation, permanently.

    Known ±1, of the same kind enqueue_depth() already documents: two `--next` in a
    row both record 1, though the second pushed the first to 2. The field is an
    ask-time depth by definition, not a promise about playback order.

    An outstanding claim counts as that one song even while current_song is still
    None: loop() sets it only after taking the prefetch result out of its slot, and
    a --next landing in that window waits behind the song about to start. Reading
    current_song alone recorded 0 there — a front insert that waited behind nothing.
    """
    return 1 if mp.current_song is not None or mp.queue.claim_outstanding() else 0


def _join_succeeded(ctx: commands.Context) -> bool:
    """Did the join a cold-start command just ran leave a USABLE voice client?

    is_connected(), not just the type: discord.py registers the client on the guild
    BEFORE the handshake completes, and vc.play() on a still-connecting one raises
    once per restored song. join also swallows its own failures, so a failed one
    arrives here as an absent client rather than an exception. Shared by -play and
    -resume — the two checks must never diverge, since a type-only check is exactly
    the bug this guards.
    """
    vc = ctx.voice_client
    return isinstance(vc, discord.VoiceClient) and vc.is_connected()


def _play_takes_the_queue(
    ctx: commands.Context, voice_client: Optional[discord.VoiceClient]
) -> bool:
    """Whether this -play decides what a channel hears next, rather than adding to
    the end of what it is already hearing.

    `--now` stops the current song; `--next` takes the front of the queue. Both are
    queue control, which every command other than -play is gated on — the exemption
    exists because appending costs the listeners in another channel nothing, and
    neither of these appends.

    Reads the PARSED argument: Command.prepare() runs _parse_arguments before
    call_before_hooks, so ctx.kwargs is filled by the time the gate runs. Other
    commands carry no `url` and fall out at the `.get`.

    Deliberately conservative. A paused voice client counts even though the body
    also requires a current song, because the gate cannot ask for one without
    building a player — and a wrongly-refused command costs a re-send from the right
    channel, where a wrongly-allowed one cannot be undone by the people it affected.
    """
    if voice_client is None:
        return False
    if voice_client.is_paused():
        return True
    return split_play_args(str(ctx.kwargs.get("url", ""))).mode is not PlayMode.NORMAL


def _check_voice_permissions(
    author: Union[discord.Member, discord.User],
    voice_client: Optional[discord.VoiceClient],
    command_name: str,
    *,
    queue_control: bool = False,
) -> Optional[str]:
    """Returns an error message string if validation fails, None if OK.

    -play alone is exempt from the same-channel rule: queueing into a session
    running elsewhere costs its listeners nothing, they just hear the song later.
    QUEUE CONTROL is not that — `--now` stops what that channel is hearing right
    now, and `--next` decides what it hears when the current song ends — so it is
    gated like every other order-changing command (-skip, -shuffle, -remove, -clear)
    even when it arrives as -play. Without the flag, a member in another channel
    could cut the line for a room they are not in.
    """
    if isinstance(author, discord.User):
        return f"You must be a member of this channel {author}"
    if not author.voice or not author.voice.channel:
        return f"You are not connected to a voice channel, you silly baka {author}"
    if (
        (command_name != "play" or queue_control)
        and voice_client is not None
        and voice_client.channel != author.voice.channel
    ):
        return f"Bot is already being used in channel {voice_client.channel}"
    return None


class MusicBot(commands.Cog):
    """
    class for music bot
    """

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # HACK: getattr() hides MusicBot's real dependency on MusicBotApp.redis.
        # `bot` is typed commands.Bot but `redis` is a MusicBotApp attribute, so the
        # access is spelled getattr() to quiet the type checker. getattr() returns
        # Any, downgrading Optional[aioredis.Redis] to an unchecked assertion:
        # renaming MusicBotApp.redis silently degrades every guild to no-Redis — no
        # persistence, no crash recovery — with nothing going red. Fix: type `bot` as
        # "MusicBotApp" under `if TYPE_CHECKING` (src/main.py uses that idiom to break
        # this cycle), or a Protocol carrying `redis: Optional[aioredis.Redis]`;
        # history_archive below is the same HACK and the same fix covers both.
        self.redis: Optional[aioredis.Redis] = getattr(bot, "redis", None)
        # The play-history archive's read surface. Present exactly when the archive
        # is enabled: setup_hook builds it (requiring POSTGRES_URL) before
        # load_extension constructs this cog, and leaves it None otherwise. Typed as
        # ArchiveReader — -ping's Postgres row and -leaderboard's aggregate are all
        # this class does with it.
        self.history_archive: Optional[ArchiveReader] = getattr(
            bot, "history_archive", None
        )
        # Spotify is optional: only build the client when credentials are present. When
        # None, playing a Spotify link raises SpotifyDisabledError; every other source
        # (YouTube, SoundCloud, search) is unaffected. See _require_spotify.
        self.spotify: Optional[Spotify] = (
            Spotify(redis=self.redis) if spotify_enabled() else None
        )
        # Enabled when credentials are present, disabled when absent; cog_load()
        # then probes the live API and downgrades to invalid if they don't
        # authenticate. The optimistic start is safe because cog_load runs inside
        # setup_hook, before the gateway connects — no command can arrive first.
        self._spotify_status: SpotifyStatus = (
            SpotifyStatus.ENABLED
            if self.spotify is not None
            else SpotifyStatus.DISABLED
        )
        self.mps: dict[int, MusicPlayer] = {}
        # (guild id, placement) of every -play in flight. See _play_bucket.
        self._play_inflight: set[tuple[int, PlayMode]] = set()
        # id(ctx) → the in-flight command's span, otel token and start time.
        self._active_spans: dict[int, ActiveCommand] = {}
        self.voice_watchdog = VoiceWatchdog(self)
        self._restore_tasks: set[asyncio.Task] = set()
        # Debug mode: the durable per-guild choice, its read cache and the runtime
        # sampler, all owned by DebugSettings (src/debug.py). MusicContext.send and
        # MusicPlayer read this attribute directly. Named debug_settings, not debug:
        # MusicBot.debug is the -debug command and an attribute would shadow it.
        self.debug_settings = debug_mode.DebugSettings()

    async def cog_load(self) -> None:
        """Kick off Spotify credential validation without blocking startup.
        discord.py awaits this inside setup_hook, before the bot connects, so
        anything awaited here delays it. The probe is a live network call, spawned
        fire-and-forget; _spotify_status stays optimistically enabled meanwhile."""
        # At load, not only on toggles — RuntimeSampler.apply's docstring has the
        # reason. The hydration re-syncs it once the stored choices land.
        self.debug_settings.sync_sampler()
        spawn_background(self._hydrate_debug(), self._restore_tasks)
        if self.spotify is None:
            return
        spawn_background(self._validate_spotify_credentials(), self._restore_tasks)

    async def cog_unload(self) -> None:
        """Stop the runtime sampler unconditionally. A reload that left it running
        would drip /proc reads for the life of the process. The shared Prometheus
        and Spotify sessions go with it, or a reload leaks a connector per load.

        The background tasks go FIRST: the hydration ends in sync_sampler, so one
        still in flight would restart the sampler straight after aclose() and leak
        it, holding a dead cog alive.

        Every step is guarded individually, like MusicBotApp.close(): Cog._eject
        only logs what this raises and BotBase.close swallows it, so an early
        failure would silently skip the rest and nothing would say so.
        """
        spotify = self.spotify

        async def _cancel_background() -> None:
            await asyncio.gather(*(cancel_task(t) for t in list(self._restore_tasks)))

        # Callables, not coroutines: building them up front would schedule the
        # gather before its turn and leave the rest un-awaited on any early exit,
        # which pytest's filterwarnings=error turns into a failure.
        steps: list[tuple[str, Callable[[], Awaitable[Any]]]] = [
            ("background tasks", _cancel_background),
            ("runtime sampler", self.debug_settings.aclose),
            ("prometheus session", debug_mode.close_prometheus_session),
        ]
        if spotify is not None:
            steps.append(("spotify session", spotify.aclose))
        for name, step in steps:
            try:
                await step()
            except Exception as e:
                log.warning(f"cog unload step failed ({name}): {e}")

    async def _validate_spotify_credentials(self) -> None:
        """Background credential probe (spawned by cog_load, never awaited). Only
        SpotifyAuthError marks Spotify invalid; network errors, timeouts and non-auth
        HTTP failures say nothing about validity, so the source stays enabled and a
        genuine problem surfaces on the first Spotify link."""
        spotify = self.spotify
        if spotify is None:  # narrowing for the type checker; cog_load already checked
            return
        try:
            await asyncio.wait_for(
                spotify.validate(SPOTIFY_TEST_TRACK_ID), timeout=10.0
            )
            self._spotify_status = SpotifyStatus.ENABLED
            log.info("Spotify credentials validated — Spotify source enabled")
        except SpotifyAuthError as e:
            self._spotify_status = SpotifyStatus.INVALID
            log.error(
                f"Spotify rejected the configured credentials ({e}); Spotify links "
                "will be declined. Check SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET."
            )
        except Exception as e:
            log.warning(
                "Could not validate Spotify credentials at startup "
                f"({type(e).__name__}: {e}); leaving Spotify enabled — a genuine "
                "credential problem will surface on the first Spotify link."
            )

    def _require_spotify(self) -> Spotify:
        """The Spotify client, or SpotifyDisabledError when the feature is off or
        its credentials failed startup validation. Call at every Spotify dispatch:
        it narrows the Optional away AND produces the user-facing error, whose text
        depends on why Spotify is unavailable."""
        if self.spotify is None or self._spotify_status is not SpotifyStatus.ENABLED:
            raise SpotifyDisabledError(self._spotify_status)
        return self.spotify

    def get_mp(self, ctx: commands.Context) -> MusicPlayer:
        """Return the guild's MusicPlayer, creating and starting one if absent."""
        assert ctx.guild is not None
        if ctx.guild.id in self.mps:
            mp = self.mps[ctx.guild.id]
            mp.set_context(ctx)
            return mp
        mp = MusicPlayer.from_context(self.bot, ctx, redis=self.redis)
        mp.start()
        self.mps[ctx.guild.id] = mp
        return mp

    @_tracer.start_as_current_span("bot.cleanup")
    async def cleanup(self, guild: discord.Guild) -> None:
        """Tear down the guild's MusicPlayer: cancel its background tasks,
        disconnect from voice, and clear persisted connection state. Safe to
        call concurrently — only the first caller for a given guild proceeds."""
        # Cancel any pending alone-disconnect timer before the atomic gate, so it
        # cannot fire after cleanup completes and attempt a second one. Never
        # cancels the caller — _countdown reaches here from inside its own task.
        self.voice_watchdog.cancel(guild.id)

        # Atomic pop: only the first caller proceeds. A concurrent call (e.g.
        # on_voice_state_update firing while stop's disconnect is in flight) gets
        # None and returns, avoiding the KeyError TOCTOU race.
        mp = self.mps.pop(guild.id, None)
        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        if mp is None:
            return
        log.info("going to cleanup/disconnect")
        # Claim the song being abandoned mid-play, before any await so the loop
        # cannot slip its iteration end into the window. Nothing else records it: it
        # left the queue at start, and clear_connection() drops the parked copy.
        pending_history = mp.claim_current_song_for_history()
        try:
            # Cancel tasks before disconnecting so the loop cannot wake and start
            # the next song between voice_client.stop() and cancellation.
            # disconnect() calls stop() internally, silencing audio below.
            teardown = [
                cancel_task(mp._prefetch_task),
                cancel_task(mp._progress_task),
                cancel_task(mp._heartbeat_task),
                cancel_task(mp._pause_debounce_task),
                cancel_task(mp._player),
                cancel_task(mp._restore_task),
            ]
            await asyncio.gather(*teardown)
            # Tasks are down, so no tick can race this. Dispose of the NP host so
            # no message keeps a bar frozen mid-song by the stop.
            await mp.retire_np_host_on_stop()
            if guild.voice_client:
                await guild.voice_client.disconnect(force=False)
            if pending_history is not None:
                # After the disconnect, never inside the teardown gather: gather
                # completes on its SLOWEST member, so Redis IO there delays the
                # silence -stop asked for — measured at 20s against an unreachable
                # host, and unbounded against one that accepts and then stalls,
                # since the pool sets no socket_timeout.
                await mp.history.add(pending_history)
            # The loop's CancelledError handler already resets presence, but only
            # if it was parked inside the block that handles it. Repeated here —
            # after the disconnect, so this guild's client no longer registers as
            # playing — so a stopped bot never advertises the song it stopped.
            await mp.update_activity(None)
            if mp.store is not None:
                # Intentional stop — clear channel IDs and now-playing state so
                # on_ready doesn't try to recover this guild after a restart.
                await mp.store.clear_connection()
                await mp.store.refresh_ttl()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            record_span_error(trace.get_current_span(), e)
            log.error(f"cleanup error: {type(e).__name__}: {e}", exc_info=True)

    async def cog_before_invoke(self, ctx: commands.Context) -> None:
        """discord.py hook run before every command: binds log context, opens
        the per-command trace span, and persists the invoking text channel."""
        from structlog.contextvars import bind_contextvars

        bind_contextvars(
            guild_id=str(ctx.guild.id) if ctx.guild else "none",
            user_id=str(ctx.author.id),
            command=ctx.command.name if ctx.command else "unknown",
        )

        cmd_name = ctx.command.name if ctx.command else "unknown"
        span = _tracer.start_span(
            f"command.{cmd_name}",
            attributes={
                "discord.guild_id": str(ctx.guild.id) if ctx.guild else "",
                "discord.user_id": str(ctx.author.id),
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._active_spans[id(ctx)] = ActiveCommand(
            span=span, token=token, started=time.monotonic()
        )

        try:
            if ctx.guild is None:
                return
            # -debug is observation-only (see debug.py's module docstring), and
            # get_mp() below CREATES a player. Letting it run would make the snapshot
            # report a player it just manufactured — `player no` would be unreachable
            # — and would spawn _restore_state() plus a 300s gate timeout that ends in
            # cleanup() on a guild that was doing nothing.
            #
            # Read off the command rather than compared to a literal: a hardcoded
            # "debug" here plus the same literal in its test means renaming the
            # command leaves both green while the exemption silently stops applying.
            if ctx.command is not None and ctx.command.extras.get("observation_only"):
                return
            old_channel = (
                self.mps[ctx.guild.id]._channel if ctx.guild.id in self.mps else None
            )
            mp = self.get_mp(ctx)
            if (
                isinstance(ctx.channel, discord.TextChannel)
                and old_channel != ctx.channel
                and mp.store is not None
                and ctx.guild is not None
            ):
                vc = ctx.guild.voice_client
                if isinstance(vc, discord.VoiceClient) and vc.channel is not None:
                    await mp.store.set_connection(vc.channel.id, ctx.channel.id)
        except Exception as e:
            # cog_after_invoke won't fire if cog_before_invoke raises — end span now.
            self._active_spans.pop(id(ctx))
            span.record_exception(e)
            span.set_status(StatusCode.ERROR, "before_invoke failed")
            span.end()
            otel_context.detach(token)
            raise

    async def cog_after_invoke(self, ctx: commands.Context) -> None:
        """discord.py hook run after every command: clears log context and
        closes the per-command trace span opened by cog_before_invoke."""
        from structlog.contextvars import clear_contextvars

        clear_contextvars()
        active = self._active_spans.pop(id(ctx), None)
        if active:
            active.span.end()
            otel_context.detach(active.token)

    @contextlib.asynccontextmanager
    async def traced_help(self, ctx: commands.Context) -> AsyncGenerator[None]:
        """The span cog_before_invoke would have opened, for the help paths it does
        not cover: discord.py owns the help command, and MusicBotApp.invoke's
        `--help` short-circuit bypasses dispatch. Without it a help embed's debug
        footer has no trace id and no elapsed time. Same key and bookkeeping as the
        cog hooks, so MusicContext.send finds it.
        """
        span = _tracer.start_span(
            "command.help",
            attributes={
                "discord.guild_id": str(ctx.guild.id) if ctx.guild else "",
                "discord.user_id": str(ctx.author.id),
            },
        )
        token = otel_context.attach(trace.set_span_in_context(span))
        self._active_spans[id(ctx)] = ActiveCommand(
            span=span, token=token, started=time.monotonic()
        )
        try:
            yield
        finally:
            active = self._active_spans.pop(id(ctx), None)
            if active is not None:
                active.span.end()
                otel_context.detach(active.token)

    async def cog_command_error(self, ctx: commands.Context, error: Exception) -> None:
        """discord.py hook run when a command raises: records the error on the
        active span and, for errors with no other user-visible output, notifies the user.
        """
        # Peek, don't pop: cog_after_invoke runs after this and ends the span.
        active = self._active_spans.get(id(ctx))
        if active:
            active.span.record_exception(error)
            active.span.set_status(StatusCode.ERROR, str(error))

        # validate_commands sends its own message before raising CommandError, so
        # only handle errors that produce no user-visible output.
        if isinstance(error, commands.MissingRequiredArgument):
            cmd = ctx.command
            usage = f"`{ctx.prefix}{cmd.name} {cmd.signature}`" if cmd else ""
            await ctx.send(
                embed=notice_embed(
                    f"Missing argument: `{error.param.name}`."
                    + (f" Usage: {usage}" if usage else ""),
                    discord.Color.red(),
                )
            )
        elif isinstance(error, commands.FlagError):
            # Flag parsing fails before the command body runs, so its
            # try/except never sees it — e.g. `-history --limit abc`.
            await ctx.send(
                embed=notice_embed(f"Invalid flags: {error}", discord.Color.red())
            )
        elif isinstance(error, commands.MaxConcurrencyReached):
            # Raised in prepare(), before the body, so the command's own try/except
            # never sees it (e.g. a second -ping while one is live). Worded off the
            # command name since any future guarded command lands here.
            cmd = ctx.command.name if ctx.command else "command"
            await ctx.send(
                embed=notice_embed(
                    f"A `{cmd}` request is already running in this server.",
                    discord.Color.orange(),
                )
            )

    async def validate_commands(self, ctx: commands.Context) -> None:
        """before_invoke hook: rejects the command with a user-facing message
        if the author isn't in a usable voice channel."""
        vc = ctx.voice_client
        voice_client = vc if isinstance(vc, discord.VoiceClient) else None
        command_name = ctx.command.name if ctx.command is not None else ""
        msg = _check_voice_permissions(
            ctx.author,
            voice_client,
            command_name,
            queue_control=_play_takes_the_queue(ctx, voice_client),
        )
        if msg:
            await ctx.send(embed=notice_embed(msg, discord.Color.red()))
            raise commands.CommandError(msg)

    async def _command_error(
        self,
        ctx: commands.Context,
        e: Exception,
        title: str = "Command failed",
        detail: Optional[str] = None,
    ) -> None:
        # The failure log lives here so 15 command bodies don't each repeat it.
        # exc_info=True still captures the live traceback — this runs inside the
        # command's `except`, so sys.exc_info() is `e`. The name comes from ctx
        # rather than being hand-typed.
        cmd = ctx.command.name if ctx.command else "command"
        log.error(f"{cmd} failed: {type(e).__name__}: {e}", exc_info=True)
        span = trace.get_current_span()
        record_span_error(span, e)  # full detail always goes to the span/logs
        # A caller-supplied detail wins: rendering the exception is safe for
        # user-input failures, but a command whose exceptions come from
        # infrastructure would publish what the operator sees — a DSN host and
        # port, or a runbook naming a just recipe — to whoever ran it.
        if detail is None:
            if isinstance(
                e,
                (
                    ExtractionError,
                    PlaylistInputError,
                    SpotifyRateLimitError,
                    SpotifyRequestError,
                ),
            ):
                # Show the user-safe line, not the raw message: yt-dlp's carries
                # bug-report boilerplate, a bad playlist link needs to name the
                # numbers, and a rate-limit needs to say "wait" rather than name
                # an endpoint. See each class's user_message.
                detail = e.user_message
            else:
                detail = f"**{type(e).__name__}:** {e}"
        await send_embed(
            ctx,
            title,
            detail,
            discord.Color.red(),
            footer=trace_footer(span),
        )

    @_tracer.start_as_current_span("bot.queue_source")
    async def queue_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        *,
        analytics: Analytics,
        origin: str,
    ) -> Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]:
        """Resolve a parsed URL/search source into something enqueueable: a
        ResolvedSpotifyPlaylist (titles still needing per-title YouTube resolution),
        a ResolvedYoutubePlaylist (already resolved), or a bare QueueObject.

        `analytics` is the command's ask-time head value, minted at dispatch;
        playlist tracks derive their per-track positions from it. `origin` is the
        raw command argument, carried onto every resulting item — for a collection
        the link, not the per-track search its expansion generated."""
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            # Titles, not QueueObjects — _enqueue_playlist mints the YTSources
            # they become, carrying this command's analytics.
            return ResolvedSpotifyPlaylist(
                await self._require_spotify().playlist(source.id)
            )
        elif isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            if source.list_id is None:
                raise ValueError("YTSource with type=PLAYLIST must have list_id set")
            tracks = await YTDL.yt_playlist(
                source.playlist_url,
                ctx.author,
                query_source=query_source_of(source),
                analytics=analytics,
                user_input=origin,
            )
            tracks, skipped = _apply_playlist_index(tracks, source.index)
            _apply_playlist_timestamp(tracks, source)
            return ResolvedYoutubePlaylist(tracks, skipped=skipped)
        else:
            ts: Optional[int] = None
            search: str
            if isinstance(source, SpotifySource):
                search = await self._require_spotify().track(source.id)
            elif isinstance(source, YTSource):
                search = source.ytsearch or source.url or ""
                ts = source.ts
            elif isinstance(source, SoundcloudSource):
                search = source.url
            else:
                assert_never(source)
            return await YTDL.yt_source(
                ctx.author,
                search,
                ts=ts,
                redis=self.redis,
                query_source=query_source_of(source),
                analytics=analytics,
                user_input=origin,
            )

    @_tracer.start_as_current_span("bot.enqueue_playlist")
    async def _warm_front_track(
        self, tracks: Sequence[QueueItem], placement: Placement
    ) -> None:
        """Warm the stream URL of a playlist's first track when it is about to play.

        Bulk enqueues pass prefetch=False, because N concurrent extractions mint
        URLs that expire long before playback reaches them. That is right for the
        tail and wrong for the head under `--next`: queue_put_next has just killed
        the loop's one-ahead prefetch, and the loop does not spawn another until its
        next iteration — so the song the user was promised would play next reaches
        the handoff with nothing cached and pays a full in-band extraction, which is
        the 1-4s of dead air the three-phase pipeline exists to remove.

        The head ONLY, for exactly the reason the rest are skipped. A lazy Spotify
        entry has no URL to warm yet (it resolves at dequeue), so it is left alone.
        """
        if placement is not Placement.NEXT or not tracks:
            return
        head = tracks[0]
        if isinstance(head, QueueObject):
            await YTDL.prefetch_stream(head, redis=self.redis)

    async def _enqueue_playlist(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        qobj: Union[ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist],
        mp: MusicPlayer,
        *,
        analytics: Analytics,
        origin: str,
        placement: Placement = Placement.TAIL,
    ) -> None:
        """Queue a resolved playlist and notify the channel — branches on the
        resolved shape since Spotify playlists arrive as titles needing YouTube
        search resolution while YouTube playlists arrive pre-resolved."""
        # A playlist front-inserts in full, in order, under either flag. Neither
        # front placement interrupts anything: COLD_FRONT has no current song at
        # all, and NEXT lets the one playing finish.
        #
        # NEXT takes queue_put_next, not queue_put_front: with a song playing, the
        # loop's prefetch holds a claim a plain front-insert would land behind, and
        # the playlist would start one song late. COLD_FRONT cannot be in that
        # state — the gate is held shut across the insert, so no iteration has run
        # to spawn a prefetch.
        enqueue = {
            Placement.TAIL: mp.queue_put,
            Placement.COLD_FRONT: mp.queue_put_front,
            Placement.NEXT: mp.queue_put_next,
        }[placement]
        warning = timestamp_warning(source)
        warning_line = f"\n\n{warning}" if warning else ""
        # Said, not implied: "Queued playlist" reads as "at the back" to anyone who
        # has used the command without a flag. Appended rather than woven in, so
        # both branches' titles keep the shape they already had.
        next_suffix = " — plays next" if placement is Placement.NEXT else ""
        if isinstance(qobj, ResolvedSpotifyPlaylist):
            titles = qobj.titles
            qobjs_yt = spotify_playlist_to_ytsearch(
                titles, analytics=analytics, origin=origin
            )
            log.info(f"ytsearch qobjs: {qobjs_yt}")
            shown_titles = queue_message([safe_label(t, _ECHO_ROW_MAX) for t in titles])
            await asyncio.gather(
                send_embed(
                    ctx,
                    "Queued playlist" + next_suffix,
                    f"Requested by: [{ctx.author.mention}]\n\n{shown_titles}"
                    f"{warning_line}",
                    discord.Color.blue(),
                ),
                enqueue(qobjs_yt, prefetch=False),
                self._warm_front_track(qobjs_yt, placement),
                ctx.message.add_reaction("👍"),
            )
        else:
            # HACK: this assert stands in for a correlation the signature cannot
            # express — `source` and `qobj` are separate parameters, but a
            # ResolvedYoutubePlaylist always arrives with a YTSource. `python -O`
            # strips it, leaving the attribute reads below unguarded. Fix: have the
            # Resolved*Playlist dataclasses carry their own source.
            assert isinstance(source, YTSource)
            playlist_url = source.playlist_url
            # Mirrors the Spotify branch: a YTSource playlist resolves via
            # yt_playlist() to fully-formed QueueObjects.
            tracks = qobj.tracks
            count = len(tracks)
            log.info(f"yt playlist track count: {count}")
            # Stated, not silent: the user pasted a link and got fewer songs than
            # the playlist holds, and only the `index=` in their own URL explains
            # it.
            skipped_line = (
                f"Starting at #{qobj.skipped + 1} — skipped {qobj.skipped} "
                f"earlier {pluralize(qobj.skipped, 'song')}\n"
                if qobj.skipped
                else ""
            )
            shown_titles = queue_message(
                [safe_label(q.title, _ECHO_ROW_MAX) for q in islice(tracks, 10)]
            )
            await asyncio.gather(
                send_embed(
                    ctx,
                    f"Queued playlist — {count} "
                    f"{pluralize(count, 'song')}{next_suffix}",
                    f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n"
                    f"{skipped_line}\n{shown_titles}{warning_line}",
                    discord.Color.blue(),
                ),
                enqueue(tracks, prefetch=False),
                self._warm_front_track(tracks, placement),
                ctx.message.add_reaction("👍"),
            )

    async def _send_playing_next(
        self,
        ctx: commands.Context,
        qobj: QueueObject,
        *,
        note: str,
        reaction: str = "👍",
    ) -> None:
        """The "Playing next" confirmation.

        One builder for the two paths that make that promise — `-play --next`, and
        the interjection whose song ended before it could be interrupted — because
        they are the same claim about the same queue position, and a copy that
        drifted between them would describe the same insert two ways. `note` is the
        only difference: why this song is next.
        """
        await asyncio.gather(
            send_embed(
                ctx,
                truncate_embed_title(f"▶️ Playing next: {qobj.title}"),
                f"Requested by: [{ctx.author.mention}]\n{note}",
                discord.Color.blue(),
                thumbnail=qobj.thumbnail,
            ),
            ctx.message.add_reaction(reaction),
        )

    @_tracer.start_as_current_span("bot.enqueue_single")
    async def _enqueue_single(
        self,
        ctx: commands.Context,
        qobj: QueueObject,
        mp: MusicPlayer,
        *,
        placement: Placement = Placement.TAIL,
        note: str = "",
        warning: Optional[str] = None,
    ) -> None:
        """`warning` rides the confirmation embed when there is one. Every exit
        below sends it either way: the embed is conditional (a song that starts
        immediately gets none) and the warning is about what the user typed, so
        losing it on the quietest path would hide it in the common case of
        queueing the first song."""
        vc = ctx.voice_client
        if placement is Placement.COLD_FRONT:
            # The "Est. playing at" embed below would be wrong: a restored queue is
            # non-empty but its entries sit BEHIND this song. The resume notice
            # replaces it — it names the song starting now (nothing else does; the
            # gate is shut, so there is no NP block to host). Built before the
            # insert, while the queue holds only the restored entries.
            resume_notice = mp.build_resume_notice_embed(qobj)
            coros: list[Coroutine[Any, Any, Any]] = [
                mp.queue_put_front(qobj),
                ctx.message.add_reaction("👍"),
            ]
            if resume_notice is not None:
                coros.append(ctx.send(embed=resume_notice))
            if warning is not None:
                coros.append(
                    ctx.send(embed=notice_embed(warning, discord.Color.orange()))
                )
            await asyncio.gather(*coros)
            log.info(f"play (front) qsize: {mp.queue.qsize()}")
            return

        if placement is Placement.NEXT:
            # No "Est. playing at": estimated_playing_at() walks the whole queue and
            # seeds from the current song's FULL duration as a proxy for what is
            # left of it — fine at the back, badly wrong for the very next slot. A
            # precise one computed from position_secs would give this reply a
            # different clock from the progress bar, -queue and the presence line,
            # which is a bug this repo has had once already. Naming the song it
            # waits behind is exact and needs no new arithmetic.
            coros: list[Coroutine[Any, Any, Any]] = [
                mp.queue_put_next(qobj),
                self._send_playing_next(ctx, qobj, note=_plays_after_note(mp, vc)),
            ]
            if warning is not None:
                coros.append(
                    ctx.send(embed=notice_embed(warning, discord.Color.orange()))
                )
            await asyncio.gather(*coros)
            log.info(f"play (next) qsize: {mp.queue.qsize()}")
            return

        # A note is never dropped: it is the only word the user gets about tracks
        # queued behind this one, and the queue being empty is not a reason to
        # withhold it.
        should_show_queued = (
            bool(note)
            or mp.queue.qsize() > 0
            or (isinstance(vc, discord.VoiceClient) and vc.is_playing())
        )
        coros: list[Coroutine[Any, Any, Any]] = [
            mp.queue_put(qobj),
            ctx.message.add_reaction("👍"),
        ]
        if should_show_queued:
            warning_line = f"\n\n{warning}" if warning else ""
            coros.append(
                send_embed(
                    ctx,
                    "Queued song",
                    (
                        f"Requested by: [{ctx.author.mention}]\n"
                        f"{qobj.title} - ({qobj.webpage_url})\n"
                        f"Est. playing at {mp.estimated_playing_at()}{note}"
                        f"{warning_line}"
                    ),
                    discord.Color.blue(),
                    thumbnail=qobj.thumbnail,
                )
            )
        elif warning is not None:
            # Nothing else is being sent on this path — the song starts now and
            # the NP card speaks for it — so the warning needs its own message.
            coros.append(ctx.send(embed=notice_embed(warning, discord.Color.orange())))
        await asyncio.gather(*coros)
        log.info(f"play qsize: {mp.queue.qsize()}")

    @commands.command(
        name="play",
        aliases=["p", "sing"],
        brief="queue a song and start playing",
        usage="[--now|--next] <url|search>",
        help=(
            "Queues a song and starts playback. Accepts a YouTube link, a YouTube "
            "playlist, a Spotify track or playlist link, a SoundCloud link, or plain "
            "words to search YouTube with.\n\n"
            "If the bot is not connected yet it joins your voice channel first. "
            "Otherwise the song is appended to the queue with an estimated start time. "
            "A `?t=` / `?ts=` timestamp starts it at that offset, and a playlist link's "
            "`&index=` starts from that position instead of from the first track.\n\n"
            "One option, as the first word:\n\n"
            "`--now` plays it immediately. The interrupted song returns from the exact "
            "position it left off at, paused if it was paused, unless it was nearly "
            "over. Interrupt again and the parked songs unwind most recent first.\n\n"
            "`--next` queues it at the front instead, without interrupting anything."
            "\n\n"
            "Both take a whole playlist in full. With `--now` that means the "
            "interrupted song does not return until the last track — `-remove` with "
            "the same link takes the whole thing back out."
        ),
        extras={
            "category": "Playback",
            "examples": [
                "-play never gonna give you up",
                "-p --now never gonna give you up",
                "-p --next https://youtu.be/dQw4w9WgXcQ",
                "-play https://youtu.be/dQw4w9WgXcQ?t=43",
                "-play https://www.youtube.com/playlist?list=PLabc&index=4",
                "-play https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
                "-p https://soundcloud.com/artist/track",
            ],
            "note": (
                "Spotify links are matched to YouTube audio one title at a time, so a "
                "long playlist takes a few seconds to finish queueing — and one plain "
                "`-play` runs at a time per server, so a second one sent meanwhile is "
                "declined rather than queued. `--now` and `--next` have their own "
                "limit, so an urgent request still goes through."
            ),
        },
    )
    # Serialized per guild PER PLACEMENT by _play_bucket, not by max_concurrency:
    # the decorator acquires before the argument is parsed, so it cannot see which
    # placement was asked for.
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.play")
    async def play(self, ctx: commands.Context, *, url: str) -> None:
        # Consume-rest, so a multi-word search arrives whole. It is what -remove
        # matches on, and a positional would bind only "never" out of "never gonna
        # give you up" — making that removable by a prefix nobody typed. parse_input
        # reads the search off THIS argument, not off the raw message, so whatever
        # is passed here is the whole input as far as everything downstream knows.
        # discord.py already strips a consume-rest argument; this covers the direct
        # callers (tests) that never went through its parser.
        args = split_play_args(url.strip())
        async with self._play_bucket(ctx, args.mode):
            await self._play(ctx, args)

    @contextlib.asynccontextmanager
    async def _play_bucket(
        self, ctx: commands.Context, mode: PlayMode
    ) -> AsyncGenerator[None, None]:
        """One -play in flight per guild PER PLACEMENT, declining rather than
        queueing — max_concurrency's contract, with a key it cannot express.

        TWO buckets, because -playnow used to be its own command and therefore had
        its own semaphore: discord.py attaches a MaxConcurrency instance per
        Command, so folding it into a flag silently put the urgent path behind the
        ordinary one. A -play resolving a large playlist holds its bucket for the
        whole flat extraction — measured at 99.3s on a 5,547-track link — and for
        all of it `-p --now` was answered "already running" instead of interrupting.

        Sharing one bucket within a placement is the point: two concurrent
        interjections both read a live current_song and both park a resume tail for
        it, so one play comes back twice.

        Not max_concurrency itself because prepare() acquires before
        _parse_arguments, so the flag is not known yet. MaxConcurrencyReached is
        raised by hand for the same reason it is caught by hand — cog_command_error
        already renders it, and the user-facing wording should not fork.
        """
        key = (ctx.guild.id if ctx.guild else 0, mode)
        if key in self._play_inflight:
            raise commands.MaxConcurrencyReached(1, commands.BucketType.guild)
        # No await between the check and the claim, so the pair is atomic on one
        # event loop and two invocations in the same tick cannot both pass.
        self._play_inflight.add(key)
        try:
            yield
        finally:
            self._play_inflight.discard(key)

    async def _play(self, ctx: commands.Context, args: PlayArgs) -> None:
        """The body behind -play. Takes the argument already split, so the caller
        decides whether this is an interjection."""
        trace.get_current_span().set_attribute("play.mode", args.mode.value)
        # ONE rebind, so every `origin=url` below is the query the user typed with
        # the flag off it. Threading args.query per call site instead would leave a
        # fourth to miss, and a leaked flag persists a user_input -remove cannot
        # match — silently, since nothing else reads that field.
        # read_rest hands the quotes through, and a quoted origin is one
        # -remove would have to match literally.
        url = unquote_argument(args.query)
        # The error title names the branch, not the flag: `-p --now x` on an idle
        # bot queues like any other -play, and "failed to play song now" would
        # describe an interjection that never happened.
        interjecting = False
        async with background_typing(ctx):
            try:
                if args.dash_typo is not None:
                    await ctx.send(
                        embed=notice_embed(
                            f"Did you mean `{args.dash_typo}`? Options take two "
                            "dashes.",
                            discord.Color.orange(),
                        )
                    )
                    return
                if not url:
                    await ctx.send(
                        embed=notice_embed(
                            f"Missing argument: `url`. Usage: `{ctx.prefix}play "
                            f"[{NOW_FLAG}|{NEXT_FLAG}] <url|search>`",
                            discord.Color.red(),
                        )
                    )
                    return

                # Bound before the join below, not after: every failure path hands
                # this exact player to _abandon_cold_start, and a get_mp() issued
                # after its cleanup() would build and start a fresh one. Free here —
                # cog_before_invoke already created it before this body ran.
                mp = self.get_mp(ctx)
                vc = ctx.voice_client
                # A narrowed Optional, NOT a bool: VoiceProtocol carries neither
                # is_playing nor is_paused, so a `live: bool` leaves vc unnarrowed
                # at both use sites and pyright rejects the lot. Load-bearing.
                live_vc = (
                    vc
                    if isinstance(vc, discord.VoiceClient)
                    and mp.current_song is not None
                    and (vc.is_playing() or vc.is_paused())
                    else None
                )
                # Decided BEFORE the resolve, as the separate command was: whether
                # a song is live is what separates interrupting one from queueing
                # ahead of it, and the resolve is where that can change.
                if live_vc is not None:
                    if args.mode is PlayMode.NOW:
                        interjecting = True
                        return await self._interject_flow(ctx, url, mp, live_vc)
                    if live_vc.is_paused() and args.mode is not PlayMode.NEXT:
                        # Paused → interject, not append: appending leaves the bot
                        # silent with the request buried behind a paused song. The
                        # interrupted song returns PLAYING — "-play means play" —
                        # which is the one thing that differs from the --now leg.
                        #
                        # `--next` is carved out because that reasoning does not
                        # survive it: the request is not buried behind the paused
                        # song, it IS next. Interjecting would stop the song the
                        # user chose to keep, which is the opposite of what they
                        # typed. The confirmation says playback is still paused,
                        # since nothing else would.
                        interjecting = True
                        return await self._interject_flow(
                            ctx,
                            url,
                            mp,
                            live_vc,
                            resume_paused=False,
                            require_paused=True,
                        )

                source = parse_input(url)

                qobj: Union[
                    QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist
                ]
                async with contextlib.AsyncExitStack() as stack:
                    # Not connected, so this song jumps ahead of any queue restored
                    # from Redis (a -stop leaves its queue persisted). -play on a
                    # disconnected bot means "play this", not "play the leftovers".
                    #
                    # Named for the CAUSE, not the effect: this one flag decides the
                    # analytics shortcut, the join dance below, and the insert
                    # position, and only the first two are about being out of voice.
                    # The position is a separate question, which is why it is a
                    # separate `placement` value further down.
                    cold_start = not ctx.voice_client
                    if cold_start:
                        placement = Placement.COLD_FRONT
                    elif args.mode is not PlayMode.NORMAL:
                        # BOTH flags, not just --next. `--now` reaches here only
                        # when there was nothing to interrupt — connected, but no
                        # song live — and the interruption is the only part of it
                        # that needs one. Falling to TAIL there put "plays it
                        # immediately" behind the whole queue, which is strictly
                        # worse than --next in the same state; it is reachable for
                        # the length of every resolve, and for a restored queue's
                        # first song after a restart.
                        placement = Placement.NEXT
                    else:
                        placement = Placement.TAIL
                    # Ask-time analytics, read ONCE at dispatch: the command
                    # message's snowflake time, so the wait covers gateway
                    # delivery and the resolve below. Cold ⇒ depth 0, the
                    # cold-start song plays ahead of the restored queue.
                    if cold_start:
                        position = 0
                    elif placement is Placement.NEXT:
                        # No restore wait for the depth here — this number does not
                        # come from the queue at all, so a queue still replaying
                        # cannot make it wrong.
                        position = _front_insert_depth(mp)
                    else:
                        # Wait out any in-flight restore: the queue stays empty
                        # until restore_entries() replays it, so a -play in the
                        # crash-recovery window would read 0 behind a queue about
                        # to reappear. Already set in the common case.
                        await mp.wait_for_restore(timeout=DEPTH_RESTORE_WAIT_SECS)
                        position = mp.enqueue_depth()
                    analytics = Analytics(
                        queued_at=ctx.message.created_at.timestamp(),
                        queue_position=position,
                    )
                    if cold_start:
                        # Hold the gate across the join below: join opens it the
                        # moment the handshake lands, which would start the
                        # restored head while queue_source is still extracting.
                        # Released on exiting the stack, after the front insertion.
                        await stack.enter_async_context(mp.defer_playback())
                        # Concurrent with queue_source: both are pure I/O (voice
                        # handshake vs yt-dlp extraction) with no data dependency.
                        # Awaiting join_task after queue_source guarantees the voice
                        # client is ready before queue_put fires.
                        join_task = asyncio.create_task(ctx.invoke(self.join))
                        try:
                            qobj = await self.queue_source(
                                ctx, source, analytics=analytics, origin=url
                            )
                            await join_task
                        except BaseException:
                            if not join_task.done():
                                join_task.cancel()
                                with contextlib.suppress(
                                    asyncio.CancelledError, Exception
                                ):
                                    await join_task
                            # Full cleanup, not just disconnect: cog_before_invoke
                            # already started a MusicPlayer's loop(), which would
                            # zombie for up to 300s on queue.get() with
                            # clear_connection() never firing — spurious crash
                            # recovery on restart.
                            await self._abandon_cold_start(ctx, mp)
                            raise
                        # Inserting onto a join that produced no usable client hands
                        # the loop a song it can only raise on.
                        if not _join_succeeded(ctx):
                            await self._abandon_cold_start(ctx, mp)
                            return
                    else:
                        qobj = await self.queue_source(
                            ctx, source, analytics=analytics, origin=url
                        )

                    log.info(f"Voice client: {ctx.voice_client}")

                    if placement is not Placement.TAIL:
                        # Order matters: put_front LPUSHes the mirror while
                        # restore_entries replays already-listed entries in memory
                        # only, so inserting first double-queues this song. A restore
                        # that never lands is therefore a reason NOT to insert — and
                        # the wait is bounded, since the pool sets no socket_timeout.
                        #
                        # Every front insert, not just the cold one: a warm `--next`
                        # is nearly always past its restore (a playing song proves
                        # the loop cleared _restore_complete), but connected-and-idle
                        # during crash recovery is exactly the window this guards,
                        # and the warm path's own 1s wait ignores its result because
                        # only an analytics integer rides on it.
                        if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                            # Cold start ONLY. _abandon_cold_start cancels the
                            # player's tasks and disconnects it, which is right for a
                            # join that never landed and catastrophic on a warm
                            # player — it would stop the music over a Redis blink.
                            if cold_start:
                                await self._abandon_cold_start(ctx, mp)
                            await ctx.send(
                                embed=notice_embed(
                                    "Couldn't reach this server's saved queue, so "
                                    "your song wasn't queued — try again in a "
                                    "moment.",
                                    discord.Color.red(),
                                )
                            )
                            return

                    if isinstance(qobj, QueueObject):
                        await self._enqueue_single(
                            ctx,
                            qobj,
                            mp,
                            placement=placement,
                            warning=timestamp_warning(source),
                        )
                    else:
                        await self._enqueue_playlist(
                            ctx,
                            source,
                            qobj,
                            mp,
                            placement=placement,
                            analytics=analytics,
                            origin=url,
                        )

            except Exception as e:
                await self._command_error(
                    ctx,
                    e,
                    title="Failed to play song now"
                    if interjecting
                    else "Failed to queue song",
                )

    async def _resolve_interjection_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        *,
        origin: str,
    ) -> tuple[QueueObject, list[QueueItem]]:
        """Resolve an interjection's input into (head, everything behind it).

        Two values rather than one list because the head is not interchangeable
        with the rest: it has to be a resolved QueueObject to be interrupted with,
        while the tail may hold lazy YTSources. Returning one list would need an
        `assert isinstance` at the call site to say so, and `python -O` strips those
        (see the HACK on the playlist branch of `play`).

        The head interrupts and the rest queue behind it, so a playlist no longer
        collapses to its first track. That collapse used to exist to bound how long
        the interrupted song waited to come back; it now waits for the whole
        playlist, which is a deliberate call — `--now <playlist>` most plainly means
        "play this playlist now", and one `-remove <the link>` takes the whole thing
        back out if it was a misfire.

        Named for what it does rather than for who asks: `-play --now` is one
        caller, and -play on a paused song is the other.

        `origin` is the raw command argument, passed down by every branch — for a
        playlist it is the link, not the titles the expansion generated."""
        # Ask-time analytics: the message's snowflake time, and depth 0 for the head
        # — an interjection plays immediately. Tracks behind it derive their own
        # positions from this base (yt_playlist and spotify_playlist_to_ytsearch
        # both do), so they read 1, 2, … The caller re-mints the head's depth on the
        # two paths where it ends up queueing instead.
        analytics = Analytics(
            queued_at=ctx.message.created_at.timestamp(), queue_position=0
        )
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            titles = await self._require_spotify().playlist(source.id)
            if not titles:
                raise ValueError("Playlist has no tracks")
            yts = spotify_playlist_to_ytsearch(
                titles, analytics=analytics, origin=origin
            )
            # Only the head is resolved here — it has to be playable to interrupt
            # with. The rest stay lazy YouTube searches resolved at dequeue, which
            # is what keeps a 100-track album from paying 100 searches up front.
            # Both playlist branches resolve directly rather than through
            # queue_source, so each passes its own metadata.
            head = await YTDL.yt_source(
                ctx.author,
                yts[0].ytsearch or "",
                redis=self.redis,
                query_source=query_source_of(yts[0]),
                analytics=analytics,
                user_input=origin,
            )
            return head, list(yts[1:])
        if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            tracks = await YTDL.yt_playlist(
                source.playlist_url,
                ctx.author,
                query_source=query_source_of(source),
                analytics=analytics,
                user_input=origin,
            )
            # Indexed here too: `--now` on a link copied mid-playlist starts at the
            # track the user was looking at, not the playlist's first.
            tracks, skipped = _apply_playlist_index(tracks, source.index)
            _apply_playlist_timestamp(tracks, source)
            if skipped:
                await ctx.send(
                    embed=notice_embed(
                        f"Starting at **#{skipped + 1}** — skipped {skipped} "
                        f"earlier {pluralize(skipped, 'song')}.",
                        discord.Color.orange(),
                    )
                )
            return tracks[0], list(tracks[1:])
        qobj = await self.queue_source(ctx, source, analytics=analytics, origin=origin)
        assert isinstance(qobj, QueueObject)
        return qobj, []

    @_tracer.start_as_current_span("bot.interject_flow")
    async def _interject_flow(
        self,
        ctx: commands.Context,
        url: str,
        mp: MusicPlayer,
        vc: discord.VoiceClient,
        *,
        resume_paused: bool = True,
        require_paused: bool = False,
    ) -> None:
        """Resolve `url` to one song, interrupt what is playing, and report.

        Shared by `-play --now` and by `-play` on a paused song; they differ only
        in resume_paused (`--now` restores paused-in → paused-out, plain `-play` brings it
        back playing). require_paused re-reads the pause state after resolution,
        before committing: `-play` interjects only *because* the song is paused, so a
        `-resume` landing during the 1–4s extraction removes the reason and the track
        is appended instead. Reading it here rather than at command entry also means
        a song that fails to resolve never stops the paused song.
        """
        source = parse_input(url)
        qobj, follow_on = await self._resolve_interjection_source(
            ctx, source, origin=url
        )
        # The head only. `interjected` is attribution — which song cut the line —
        # and the tracks behind it queued normally, exactly as a plain -play's would.
        qobj.interjected = True

        # Warm the stream-URL cache before interrupting: a cache miss at dequeue puts
        # seconds of yt-dlp dead air between the interrupt and the new song. Awaited,
        # not spawned — the current song plays through the wait. No-op without Redis;
        # also back-fills duration/thumbnail for the embeds below.
        #
        # The head only, again: warming N tracks would be N concurrent extractions
        # minting URLs that expire long before playback reaches them, which is the
        # same reason bulk enqueues pass prefetch=False.
        await YTDL.prefetch_stream(qobj, redis=self.redis)

        if require_paused and not vc.is_paused():
            # Resumed during the resolve — the reason to interject is gone, so
            # append rather than interrupt a song the user just chose to keep
            # playing. Clear the marker: a normally queued song must not trigger
            # replace semantics later.
            qobj.interjected = False
            # An ordinary append now, behind the whole queue, so replace the 0
            # minted for the interjection. Read here: the queue moved during the
            # resolve.
            depth = mp.enqueue_depth()
            qobj.analytics = replace(qobj.analytics, queue_position=depth)
            note = ""
            if follow_on:
                # The playlist behind it still belongs behind it, and the head just
                # went to the tail — so these follow, not front-insert. Their
                # ask-time depths were minted for a front insert (1..N-1) and are
                # re-minted here for the same reason the head's is: they went to the
                # BACK, and play_history keeps whatever number is on them forever.
                follow_on = [
                    _with_queue_position(item, depth + offset)
                    for offset, item in enumerate(follow_on, start=1)
                ]
                await mp.queue_put(follow_on, prefetch=False)
                note = _collection_note(url, len(follow_on) + 1, head_playing=False)
            await self._enqueue_single(
                ctx, qobj, mp, note=note, warning=timestamp_warning(source)
            )
            return

        outcome = await mp.interject(
            qobj, vc, resume_paused=resume_paused, follow_on=follow_on
        )
        if outcome is None:
            # The song ended during the resolve — nothing left to interrupt. Insert
            # qobj directly rather than re-invoking -play, which would re-parse,
            # re-resolve and enqueue every track a second time. Front, not append:
            # the user asked for "now", and this window can be seconds long with
            # songs queued behind.
            # It interrupted nothing, so keeping the marker would attribute an
            # interjection that never happened.
            qobj.interjected = False
            # interject() also returns None when the loop moved on to a
            # DIFFERENT song, which this insert waits behind. One, never the
            # queue depth: it goes to the front.
            qobj.analytics = replace(
                qobj.analytics, queue_position=_front_insert_depth(mp)
            )
            # queue_put_next, not queue_put_front: the embed below promises "play
            # next", and a bare front-insert only delivers that when nothing is
            # queued — the loop's prefetch holds a claim the insert would land
            # behind. interject() returned None here without reaching its own
            # neutralize, so this is the one path that still has to do it.
            # prefetch=False — the stream URL was warmed above.
            await mp.queue_put_next([qobj, *follow_on], prefetch=False)
            note = (
                "The song being interrupted already ended — "
                "queued to play next instead."
            )
            if follow_on:
                # Nothing was interrupted, so the head is QUEUED rather than
                # playing: it counts, and -remove reaches it.
                note += _collection_note(url, len(follow_on) + 1, head_playing=False)
            await self._send_playing_next(ctx, qobj, note=note, reaction="⏯️")
            return

        if outcome.resume_position is None:
            desc = (
                f"**{outcome.interrupted_title}** was nearly finished "
                f"and will not resume."
            )
        elif outcome.returns_paused:
            # returns_paused, not was_paused: with resume_paused=False the song was
            # paused but comes back playing, so "will return paused" would lie.
            desc = (
                f"**{outcome.interrupted_title}** will return paused at "
                f"`{outcome.resume_position_str}`."
            )
        elif outcome.was_paused:
            desc = (
                f"**{outcome.interrupted_title}** was paused at "
                f"`{outcome.resume_position_str}` and will resume from there."
            )
        else:
            desc = (
                f"**{outcome.interrupted_title}** will resume at "
                f"`{outcome.resume_position_str}`."
            )
        if follow_on:
            # The whole playlist came with it, so the interrupted song now waits
            # behind all of it — on a long link, in practice forever. Said outright
            # rather than left to be discovered from -queue, and paired with the
            # undo, because a 500-track `--now` is a plausible misfire and
            # `-remove <the link>` takes those tracks back out in one command
            # (remove_matcher matches on user_input, which each track carries).
            #
            # The tail only, and head_playing: this song is playing NOW, so it has
            # no queue object for -remove to match and the count must not include it.
            desc += _collection_note(
                url,
                len(follow_on),
                returns=(
                    f" **{outcome.interrupted_title}** returns after the last of them."
                    if outcome.resume_position is not None
                    else ""
                ),
                head_playing=True,
            )
        await asyncio.gather(
            send_embed(
                ctx,
                truncate_embed_title(f"▶️ Playing now: {qobj.title}"),
                f"Requested by: [{ctx.author.mention}]\n{desc}",
                discord.Color.blue(),
                thumbnail=qobj.thumbnail,
            ),
            ctx.message.add_reaction("⏯️"),
        )

    @commands.command(
        name="skip",
        aliases=["sk"],
        brief="skip to the next song in the queue",
        help=(
            "Stops the current song and immediately starts the next one in the "
            "queue. If the queue is empty the bot stays connected and idles "
            "until you queue something else.\n\n"
            "A **paused** song can be skipped too — it is dropped and the next "
            "song starts playing."
        ),
        extras={"category": "Playback", "examples": ["-skip", "-sk"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.skip")
    async def skip(self, ctx: commands.Context) -> None:
        try:
            vc = ctx.voice_client
            if not isinstance(vc, discord.VoiceClient):
                return
            # is_playing() is False while paused, so gating on it alone made -skip
            # a total no-op on a paused song — not even the reaction.
            if not (vc.is_playing() or vc.is_paused()):
                return

            # Capture before stop(): the loop's song-end bookkeeping clears
            # current_song, and the notice must name the song actually skipped.
            # Primitives, not the object — the player thread calls cleanup() on it.
            skipped_title: Optional[str] = None
            skipped_position = ""
            assert ctx.guild is not None  # validate_commands rejects DMs before this
            # mps, not get_mp(): this must not build a player, and one lookup keeps
            # the mark and the paused read on the same object — cog_before_invoke can
            # rebuild a player mid-command.
            mp_for_stop = self.mps.get(ctx.guild.id)
            if vc.is_paused() and mp_for_stop is not None:
                song = mp_for_stop.current_song
                if song is not None:
                    skipped_title = song.title
                    # position_secs is frozen while paused: the exact leave point.
                    skipped_position = fmt_duration(int(song.position_secs))

            # Before vc.stop(): a skip inside ffmpeg's startup window otherwise looks
            # exactly like a stream that never opened.
            if mp_for_stop is not None:
                mp_for_stop.note_deliberate_stop()
            vc.stop()

            coros: list[Coroutine[Any, Any, Any]] = []
            if not ctx.invoked_parents:
                coros.append(ctx.message.add_reaction("⏭"))
            if skipped_title is not None:
                # A paused song makes no sound, so stopping it gives no audible
                # cue — unlike an ordinary skip, where the music changing is it.
                coros.append(
                    ctx.send(
                        embed=notice_embed(
                            f"⏭ Skipped **{skipped_title}** — was paused at "
                            f"`{skipped_position}`.",
                            discord.Color.blue(),
                        )
                    )
                )
            if coros:
                await asyncio.gather(*coros)
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="stop",
        aliases=["st"],
        brief="stop playback and disconnect, keeping the queue",
        help=(
            "Stops the current song, removes the Now Playing card and "
            "disconnects the bot from the voice channel.\n\n"
            "This is the full teardown — use `-pause` if you only want to take a "
            "break, or `-clear` if you want to empty the queue but keep playing.\n\n"
            "The queue is **kept** on the server for 24 hours, so `-resume` (or "
            "the next `-play`) picks it back up where it left off. The song that "
            "was playing does not come back, but it is recorded in `-history`."
        ),
        extras={"category": "Playback", "examples": ["-stop", "-st"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.stop")
    async def stop(self, ctx: commands.Context) -> None:
        try:
            # Don't skip before cleanup: skip fires voice_client.stop(), whose
            # after callback (play_next.set) gives the loop a window to start the
            # next song before it is cancelled. cleanup() cancels _player first and
            # disconnect() stops the audio subprocess, so no skip is needed.
            vc = discord.utils.get(self.bot.voice_clients, guild=ctx.guild)
            if vc is not None and ctx.guild is not None:
                await ctx.message.add_reaction("👋")
                await self.cleanup(ctx.guild)
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="pause",
        aliases=["po"],
        brief="pause the current song",
        help=(
            "Pauses playback and posts the exact position the song stopped at. "
            "The queue and the bot's voice connection are kept — `-resume` picks "
            "the song back up from that position."
        ),
        extras={"category": "Playback", "examples": ["-pause", "-po"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.pause")
    async def pause(self, ctx: commands.Context) -> None:
        try:
            vc = ctx.voice_client
            if isinstance(vc, discord.VoiceClient) and vc.is_playing():
                mp = self.get_mp(ctx)
                await mp.pause(vc)
                await ctx.message.add_reaction("⏸️")
                embed = mp.build_pause_confirmation_embed()
                if embed is not None:
                    await ctx.send(embed=embed)
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="resume",
        aliases=["r"],
        brief="resume a paused song",
        help=(
            "Resumes a paused song from the position it stopped at, and re-pins "
            "the Now Playing card — with its live progress bar — to the bottom "
            "of the channel.\n\n"
            "If the bot is not in a voice channel it joins yours first and picks "
            "the previous session's queue back up, the same auto-join `-play` does."
        ),
        extras={
            "category": "Playback",
            "examples": ["-resume", "-r"],
            "note": (
                "A `-stop` or a disconnect keeps the queue for 24 hours, so "
                "`-resume` brings it back. The song that was playing comes back "
                "too — but only after a crash; an intentional `-stop` drops it."
            ),
        },
    )
    @commands.before_invoke(validate_commands)
    # Two racing -resumes both read `voice_client is None`, so validate_commands'
    # "already being used in channel X" check fires for neither: both join and the
    # second MOVES the bot to its own author's channel. One at a time per guild.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @_tracer.start_as_current_span("bot.resume")
    async def resume(self, ctx: commands.Context) -> None:
        try:
            vc = ctx.voice_client
            if not isinstance(vc, discord.VoiceClient):
                # Not in voice: the paused song went with the voice client, but the
                # queue outlives it in Redis. Join and pick that up instead.
                await self._resume_disconnected(ctx)
                return
            if vc.is_playing():
                await ctx.send(
                    embed=notice_embed(
                        "Already playing — nothing is paused.", discord.Color.orange()
                    )
                )
                return
            if not vc.is_paused():
                # No queue advice here: this branch also covers the seconds
                # between two songs, where the queue is not empty at all.
                await ctx.send(
                    embed=notice_embed("Nothing is paused.", discord.Color.orange())
                )
                return
            mp = self.get_mp(ctx)
            await mp.resume(vc)
            await ctx.message.add_reaction("⏭️")
            # If the -pause confirmation hosts the block, re-host it so
            # "⏸️ Paused at…" becomes history instead of sitting beneath a
            # live, advancing bar for the rest of the song.
            await mp.rehost_np_after_resume()
        except Exception as e:
            await self._command_error(ctx, e)

    async def _resume_disconnected(self, ctx: commands.Context) -> None:
        """`-resume` with the bot out of voice: join the author's channel and let the
        persisted queue play again.

        A `-stop`, an eject and a crash all leave `guild:{id}:queue` intact, so there
        is something to come back to; only a crash also leaves the song that was
        playing, which restore re-queues at its position (cleanup() scrubs those
        state fields on the other two).
        """
        assert ctx.guild is not None  # validate_commands rejects DMs before this
        async with background_typing(ctx):
            mp = self.get_mp(ctx)
            if not mp.can_rejoin_cold():
                # An eject that never reached on_voice_state_update, so cleanup never
                # ran. Rejoining around it announces a resume its wedged loop cannot
                # deliver.
                await self.cleanup(ctx.guild)
                mp = self.get_mp(ctx)
            # Restore first, unlike -play: there is no extraction to hide the join
            # behind, and joining first parks the bot in a channel for an empty queue.
            if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                await ctx.send(
                    embed=notice_embed(
                        "Still loading this server's saved queue — try `-resume` "
                        "again in a moment.",
                        discord.Color.orange(),
                    )
                )
                return
            # Built before the gate opens, while the queue head is still the
            # restored one — the loop pops it out from under this.
            embed = mp.build_rejoin_resume_embed()
            if embed is None:
                # A failed read lands here too, with a queue it never filled. Saying
                # "nothing was left" would assert what it cannot know.
                detail = (
                    "Nothing to resume — no queue was left from a previous "
                    "session. Use `-play` to start one."
                    if mp.store is not None and not mp.restore_read_failed
                    else "Can't reach the queue store, so there is nothing to "
                    "resume from. Use `-play` to start a new queue."
                )
                await ctx.send(embed=notice_embed(detail, discord.Color.orange()))
                return

            # The hold -play takes across its join: without it the head starts playing,
            # and posts its NP card, before the reply explaining the join lands.
            async with mp.defer_playback():
                try:
                    await ctx.invoke(self.join)
                    joined = _join_succeeded(ctx)
                except BaseException:
                    # join swallows Exceptions, so an escape means its error REPORTING
                    # failed, or the command was cancelled. Same wreckage, same exit.
                    await self._abandon_cold_start(ctx, mp)
                    raise
                if not joined:
                    # join already told the user why; nothing to add here.
                    await self._abandon_cold_start(ctx, mp)
                    return
                await ctx.send(embed=embed)

    async def _abandon_cold_start(self, ctx: commands.Context, mp: MusicPlayer) -> None:
        """Drop the player a cold-start command (`-play`, `-resume`) was about to
        hand a voice connection to.

        defer_playback opens the gate as it unwinds whether or not the join worked,
        and a loop waking with no voice client fails its `vc` assertion once per
        restored song — draining the in-memory queue while Redis keeps every entry.
        Tearing down first makes that gate-open land on a cancelled loop, which is inert.

        The re-park FOLLOWS cleanup(): dropping the player is what loses a
        crash-recovered head, and clear_connection() HDELs the fields it writes.

        Skipped entirely while another command holds the gate — it is mid-join on this
        same player and owns the teardown call.
        """
        if mp.playback_holds > 1:  # this command's own hold, plus someone else's
            return
        if ctx.guild is not None:
            with contextlib.suppress(Exception):
                await self.cleanup(ctx.guild)
        with contextlib.suppress(Exception):
            await mp.repark_crashed_head()

    @commands.command(
        name="shuffle",
        brief="randomly reorder the queue",
        help=(
            "Randomly reorders the songs waiting in the queue. Needs at least 3 "
            "queued songs to have any effect. The song currently playing is left "
            "alone — shuffling only touches what comes after it."
        ),
        extras={"category": "Queue", "examples": ["-shuffle"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.shuffle")
    async def shuffle(self, ctx: commands.Context) -> None:
        try:
            mp = self.get_mp(ctx)
            # Like -clear and -remove: shuffle() REBUILDS the mirror from memory,
            # so running it before restore_entries() has replayed the saved queue
            # writes an unrestored deque over it and deletes the persisted entries.
            if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                await ctx.send(
                    embed=notice_embed(
                        "Still loading this server's saved queue — try again in "
                        "a moment.",
                        discord.Color.orange(),
                    )
                )
                return
            async with background_typing(ctx):
                await ctx.send(
                    embed=notice_embed("Please wait... shuffling", discord.Color.blue())
                )
                msg = await mp.queue_shuffle()
                await ctx.message.add_reaction("🔀")
                await ctx.send(embed=notice_embed(msg, discord.Color.blue()))
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="join",
        aliases=["summon"],
        brief="connect the bot to your voice channel",
        help=(
            "Connects the bot to the voice channel you are in and reports its "
            "latency. You rarely need this — `-play` and `-resume` join for you.\n\n"
            "If the bot is already playing in a different voice channel it stays "
            "there rather than abandoning that listener."
        ),
        extras={"category": "Utility", "examples": ["-join", "-summon"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.join")
    async def join(self, ctx: commands.Context) -> None:
        try:
            assert (
                isinstance(ctx.author, discord.Member) and ctx.author.voice is not None
            )
            assert ctx.guild is not None
            channel = ctx.author.voice.channel
            assert channel is not None

            if not ctx.voice_client:
                await channel.connect(timeout=10.0)
            vc = ctx.voice_client
            if isinstance(vc, discord.VoiceClient) and vc.channel != channel:
                await vc.move_to(channel)
            await ctx.guild.change_voice_state(
                channel=channel, self_mute=False, self_deaf=True
            )

            mp = self.get_mp(ctx)
            if mp.store is not None and isinstance(ctx.channel, discord.TextChannel):
                await mp.store.set_connection(channel.id, ctx.channel.id)

            # Voice is up — release the loop so a persisted queue resumes. No-op
            # while -play holds the gate: it front-inserts its song first, then
            # opens.
            mp.open_playback_gate()

            await asyncio.gather(
                ctx.message.add_reaction("👋"),
                # Not ctx.invoke(self.ping): that runs the full ~3s dashboard on
                # every join/cold-play AND skips prepare(), losing ping's
                # max_concurrency guard. Cheap one-liner only.
                send_latency_line(ctx, self.bot.latency),
            )
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="clear",
        aliases=["c"],
        brief="empty the queue",
        help=(
            "Removes every song waiting in the queue and lists what was dropped. "
            "The song currently playing keeps going — use `-skip` to move past it "
            "or `-stop` to end the session entirely."
        ),
        extras={"category": "Queue", "examples": ["-clear", "-c"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.clear")
    async def clear(self, ctx: commands.Context) -> None:
        try:
            mp = self.get_mp(ctx)
            # Destroys the Redis mirror while reading the IN-MEMORY display, so an
            # unrestored player deletes a saved queue it cannot see — including the
            # interjection tails _flush_played would have recorded. validate_commands
            # only requires the AUTHOR in voice, so a cold player reaches here.
            if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                await ctx.send(
                    embed=notice_embed(
                        "Still loading this server's saved queue — try again in "
                        "a moment.",
                        discord.Color.orange(),
                    )
                )
                return
            cleared = await mp.queue_clear()
            if not cleared:
                await ctx.send(
                    embed=notice_embed(
                        "The queue is already empty.", discord.Color.orange()
                    )
                )
                return
            description = queue_message([safe_label(t, _ECHO_ROW_MAX) for t in cleared])
            await asyncio.gather(
                ctx.message.add_reaction("🗑️"),
                send_embed(
                    ctx,
                    f"Queue cleared — {len(cleared)} {pluralize(len(cleared), 'song')} removed",
                    description,
                    discord.Color.red(),
                ),
            )
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="remove",
        aliases=["rm"],
        brief="remove queued songs by link or by what you typed",
        usage="<link or search text>",
        help=(
            "Removes every queued song matching what you give it and reports the "
            "queue positions that were dropped, followed by the updated queue.\n\n"
            "Three things match: the YouTube link shown in the **Now Playing** "
            "card, the search text you queued with, and the link you queued with "
            "— so removing a playlist link takes back out every track it added. "
            "Run it with no argument for a reminder.\n\n"
            "Links are matched as typed, so a `youtu.be` short link will not "
            "match a song queued from a full `youtube.com` one."
        ),
        extras={
            "category": "Queue",
            "examples": [
                "-remove https://www.youtube.com/watch?v=dQw4w9WgXcQ",
                "-remove never gonna give you up",
                "-remove https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
            ],
            "note": (
                "A search term removes what that exact search queued, not "
                "anything that only looks similar."
            ),
        },
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.remove")
    async def remove(
        self, ctx: commands.Context, *, needle: Optional[str] = None
    ) -> None:
        try:
            if needle is None:
                await ctx.send(
                    embed=notice_embed(
                        "`-remove <link or search text>` — removes every queued "
                        "song that matches. Give it the YouTube link from the "
                        "**Now Playing** card, or the search text or link you "
                        "queued with; a collection link removes every track it "
                        "added.",
                        discord.Color.blue(),
                    )
                )
                return
            mp = self.get_mp(ctx)
            # Destroys the Redis mirror while reading the IN-MEMORY display, so an
            # unrestored player deletes a saved queue it cannot see — including the
            # interjection tails _flush_played would have recorded. validate_commands
            # only requires the AUTHOR in voice, so a cold player reaches here.
            if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                await ctx.send(
                    embed=notice_embed(
                        "Still loading this server's saved queue — try again in "
                        "a moment.",
                        discord.Color.orange(),
                    )
                )
                return
            outcome = await mp.queue_remove(needle)
            positions = outcome.positions
            if not positions:
                await send_embed(
                    ctx,
                    "",
                    f"No queued songs found matching: {_echo(needle)}",
                    discord.Color.red(),
                )
                return
            count = len(positions)
            noun = pluralize(count, "song")
            pos_label = pluralize(count, "Position")
            # Capped by count: one -remove of a collection link drops as many
            # positions as the collection had, and a raw join passes the 1024-char
            # field limit at 227 of them — a 400 for a removal that already
            # happened.
            shown = positions[:_MAX_SHOWN_POSITIONS]
            pos_str = ", ".join(str(p) for p in shown)
            if len(positions) > len(shown):
                pos_str += f", …and {len(positions) - len(shown)} more"
            await send_embed(
                ctx,
                f"Removed {count} {noun} from the queue",
                "",
                discord.Color.orange(),
                fields=[
                    ("Matched", _field(_matched_label(outcome, needle)), False),
                    (f"{pos_label} removed", _field(pos_str), False),
                    # Titles, like -clear reports: one argument can take out a whole
                    # playlist, and there is no undo, so a bare count is not enough
                    # to tell whether it took what the user meant.
                    (
                        "Songs",
                        _field(
                            queue_message(
                                [
                                    _echo(item_label(i), _ECHO_ROW_MAX)
                                    # Sliced before the echo: queue_message keeps 10.
                                    for i in outcome.removed[:10]
                                ]
                            )
                        ),
                        False,
                    ),
                ],
            )
            await ctx.send(embed=mp.queue_embed())
            await ctx.message.add_reaction("🗑️")
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="now",
        aliases=["np", "rn", "nowplaying"],
        brief="show the song playing right now",
        help=(
            "Shows what is playing right now — title, who requested it, and a "
            "live progress bar.\n\n"
            "In the channel the bot is playing in, this re-pins the Now Playing "
            "card to the bottom of the channel instead of posting a copy that "
            "would immediately go stale. Anywhere else you get a static snapshot."
        ),
        extras={"category": "Queue", "examples": ["-now", "-np", "-nowplaying"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.now")
    async def now(self, ctx: commands.Context) -> None:
        try:
            mp = self.get_mp(ctx)
            vc = ctx.guild.voice_client if ctx.guild else None
            song = mp.current_song
            if (
                vc is not None
                and isinstance(vc, discord.VoiceClient)
                and (vc.is_playing() or vc.is_paused())
                and song is not None
            ):
                if ctx.channel.id != mp._channel.id:
                    # Outside the player's home channel: the host never leaves
                    # home, so answer HERE with a static snapshot (MusicContext's
                    # channel guard keeps it unattached).
                    await ctx.send(embed=mp._build_now_playing_embed(song))
                    return
                # Re-host the live block at the bottom (retiring the old host)
                # rather than sending a snapshot that immediately goes stale.
                if await mp.repin_now_playing():
                    return
                # Song ended between the liveness check and the repin — fall
                # through to the static/none responses instead of silence.
            if mp.play_message is not None:
                # Crash-recovery window: current_song isn't live yet but a snapshot
                # survived the restart. Static embed (no bar) until loop() starts.
                await ctx.send(embed=mp.play_message)
            else:
                await ctx.send(
                    embed=notice_embed(
                        "No songs are currently playing.", discord.Color.orange()
                    )
                )
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="history",
        aliases=["h"],
        brief="show recently played songs",
        usage="[--limit N]",
        # Interpolated, not spelled out: this copy asserts the retention window,
        # and HISTORY_MAX_LIMIT is that window (it is HISTORY_CACHE_LIMIT — see
        # the constant above). A hand-typed number here is how the previous copy
        # came to promise permanent retention months after the list was capped.
        help=(
            "Lists the songs already played in this server, most recent first.\n\n"
            f"`--limit N` controls how many are shown ({HISTORY_MIN_LIMIT}-"
            f"{HISTORY_MAX_LIMIT}, default 10). History is stored per server and "
            f"survives a bot restart, but only the newest {HISTORY_MAX_LIMIT} plays "
            f"are kept — so `--limit {HISTORY_MAX_LIMIT}` shows the whole record."
        ),
        extras={
            "category": "Queue",
            "examples": ["-history", "-h", "-history --limit 25"],
            "note": (
                f"The newest {HISTORY_MAX_LIMIT} plays are everything this command "
                "can read. If this server's host has turned on the optional "
                "long-term archive, plays are also recorded there permanently — "
                f"but `-history` always serves the {HISTORY_MAX_LIMIT}-play window."
            ),
        },
    )
    @commands.before_invoke(validate_commands)
    # One in flight per guild, wait=False, so extra invocations are declined
    # immediately rather than queueing (cog_command_error renders
    # MaxConcurrencyReached as a notice). The reason is Discord-side: -history is the
    # heaviest send in the bot — up to 8 song embeds plus the NP block — so unbounded
    # concurrent renders are how a guild rate-limits itself out of its own channel.
    # It never reaches Postgres, so it cannot contend with the drainer.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @_tracer.start_as_current_span("bot.history")
    async def history(self, ctx: commands.Context, *, flags: HistoryFlags) -> None:
        try:
            if not (HISTORY_MIN_LIMIT <= flags.limit <= HISTORY_MAX_LIMIT):
                await ctx.send(
                    embed=notice_embed(
                        f"--limit must be between {HISTORY_MIN_LIMIT} and {HISTORY_MAX_LIMIT}",
                        discord.Color.red(),
                    )
                )
                return
            mp = self.get_mp(ctx)
            entries = await mp.history.recent(flags.limit)
            if not entries:
                await ctx.send(
                    embed=notice_embed(
                        "No songs have been played yet.", discord.Color.orange()
                    )
                )
                return
            embeds = history_embeds(entries)
            # 8 per message keeps every chunk within Discord's 10-embed cap once
            # MusicContext.send prepends the ≤2-embed NP block. Each chunk goes
            # through ctx.send — never bare channel.send in the player's channel —
            # so the adopt/retire machinery walks the block down to the last chunk.
            for start in range(0, len(embeds), HISTORY_EMBEDS_PER_MESSAGE):
                await ctx.send(
                    embeds=embeds[start : start + HISTORY_EMBEDS_PER_MESSAGE]
                )
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="leaderboard",
        aliases=["lb", "top"],
        brief="top listeners and songs (long-term archive)",
        usage="[--days N]",
        help=(
            "Shows this server's top 10 listeners and top 10 songs, ranked by "
            "total listening time (song and play counts included).\n\n"
            "`--days N` limits both boards to the last N days; without it they "
            "are all-time. The numbers come from this server's long-term play "
            "archive, so they cover every song since the archive was enabled — "
            "not just the recent plays `-history` shows. A song that just "
            "finished can take a moment to appear."
        ),
        extras={
            "category": "Queue",
            "examples": ["-leaderboard", "-lb", "-leaderboard --days 30"],
            "note": (
                "Available only when this server's host has enabled the "
                "optional long-term archive."
            ),
        },
    )
    # No validate_commands: reading a leaderboard needs no voice channel (-ping
    # is the precedent). One in flight per guild, wait=False, so command spam is
    # declined rather than queued — the 60s cache bounds the rate, this bounds
    # concurrency against the pool the drainer also draws from.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @_tracer.start_as_current_span("bot.leaderboard")
    async def leaderboard(
        self, ctx: commands.Context, *, flags: LeaderboardFlags
    ) -> None:
        try:
            # Locals: ctx.guild is a property and history_archive an attribute,
            # so narrowing on either would not survive the awaits below.
            guild = ctx.guild
            archive = self.history_archive
            if guild is None:
                await ctx.send(
                    embed=notice_embed(
                        "Leaderboards are per server — use this in a server channel.",
                        discord.Color.orange(),
                    )
                )
                return
            if archive is None:
                await ctx.send(
                    embed=notice_embed(
                        "This server's host has not enabled the long-term play "
                        "archive, so there is no leaderboard data.",
                        discord.Color.orange(),
                    )
                )
                return
            if not 0 <= flags.days <= leaderboard.MAX_DAYS:
                await ctx.send(
                    embed=notice_embed(
                        f"--days must be between 1 and {leaderboard.MAX_DAYS}. "
                        "Omit it, or pass 0, for all-time.",
                        discord.Color.red(),
                    )
                )
                return
            key = leaderboard.cache_key(guild.id, flags.days, leaderboard.TOP_N)
            board = leaderboard.from_cache(
                await cache_get(self.redis, key), top_n=leaderboard.TOP_N
            )
            if board is None:
                since = time.time() - flags.days * 86400 if flags.days else 0.0
                async with background_typing(ctx):
                    board = await archive.leaderboard(
                        guild.id, leaderboard.TOP_N, since_epoch=since
                    )
                await cache_set(
                    self.redis,
                    key,
                    leaderboard.to_cache(board),
                    leaderboard.CACHE_TTL_SECS,
                )
            embed = leaderboard.build_embed(board, days=flags.days, guild=guild)
            if embed is None:
                window = (
                    f"in the last {flags.days} {pluralize(flags.days, 'day')}"
                    if flags.days
                    else "yet"
                )
                await ctx.send(
                    embed=notice_embed(
                        f"Nothing has been archived {window} — play something first!",
                        discord.Color.orange(),
                    )
                )
                return
            await ctx.send(embed=embed)
        except Exception as e:
            # Fixed copy rather than the exception text: this is the only command
            # whose failures come from infrastructure, so the default detail would
            # publish the archive's host and port, or SchemaVersionError's
            # operator runbook, to the channel. -ping reduces the same class for
            # the same reason (ping._error_detail). The trace footer still joins
            # the report to the span, and the full exception is logged there.
            await self._command_error(
                ctx,
                e,
                title="Leaderboard unavailable",
                detail="The long-term archive could not be reached. Try again in a moment.",
            )

    @commands.command(
        name="jump",
        aliases=["j"],
        brief="jump to a queue position (in development)",
        usage="<position>",
        help=(
            "Skips straight to a given position in the queue.\n\n"
            "⚠️ Not implemented yet — the command replies that it is in "
            "development. Use `-skip` to advance one song at a time."
        ),
        extras={"category": "Queue", "examples": ["-jump 3"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.jump")
    async def jump(self, ctx: commands.Context) -> None:
        try:
            # TODO: Implement -jump or remove it from the command list.
            # The help text advertises it while the body only replies "currently in
            # development", so the bot promises a feature it does not have.
            # Implementing it is a drain/rotate over GuildQueue, shaped like remove().
            await ctx.send(
                embed=notice_embed("currently in development", discord.Color.blue())
            )
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="queue",
        aliases=["q"],
        brief="list the songs waiting to play",
        help=(
            "Lists the songs waiting to play, in the order they will be played "
            "(up to 10, newest queue additions last). The queue is stored per "
            "server and is restored if the bot restarts."
        ),
        extras={"category": "Queue", "examples": ["-queue", "-q"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.queue")
    async def queue(self, ctx: commands.Context) -> None:
        try:
            mp = self.get_mp(ctx)
            await ctx.send(embed=mp.queue_embed())
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="volume",
        aliases=["v", "vol", "sound"],
        brief="set playback volume (0-100)",
        usage="<0-100>",
        help=(
            "Sets playback volume as a percentage between 0 and 100.\n\n"
            "The new level takes effect on the **next** song, not the one "
            "currently playing. It is saved per server, so it still applies "
            "after a restart."
        ),
        extras={"category": "Playback", "examples": ["-volume 50", "-vol 100"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.volume")
    async def volume(self, ctx: commands.Context, volume: str) -> None:
        try:
            try:
                volume_pct = int(volume)
            except ValueError:
                await ctx.send(
                    embed=notice_embed(
                        "Volume must be a number between 0 and 100",
                        discord.Color.red(),
                    )
                )
                return
            if not 0 <= volume_pct <= 100:
                await ctx.send(
                    embed=notice_embed(
                        "Volume must be between 0 and 100", discord.Color.red()
                    )
                )
                return
            mp = self.get_mp(ctx)
            mp.volume = volume_pct / 100
            persisted = False
            if mp.store is not None:
                persisted = await mp.store.set_volume(mp.volume)
            # Same rule as the debug toggle: the help promises this survives a
            # restart, so a write that did not land is named rather than rounded
            # up to success — a level that quietly reverts reads as being ignored.
            durability = (
                "It is saved for this server."
                if persisted
                else "⚠️ It could not be saved (Redis is unavailable), so it "
                "applies until the bot restarts."
            )
            await ctx.send(
                embed=notice_embed(
                    f"Set volume to {volume_pct}% (takes effect on next song). "
                    + durability,
                    discord.Color.blue(),
                )
            )
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="ping",
        aliases=["latency", "l", "delay", "health", "status"],
        brief="bot & dependency health + versions",
        help=(
            "Live health check: round-trip latency to Discord, Redis, Spotify, "
            "Postgres and the OTEL collector, plus the running bot / yt-dlp / "
            "ffmpeg versions. The message posts instantly and fills in as each "
            "dependency answers; the embed colour tracks the worst one. The "
            "Spotify row also reports the optional source's state: not "
            "configured, or configured with credentials Spotify rejected."
        ),
        extras={"category": "Utility", "examples": ["-ping", "-health"]},
    )
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @_tracer.start_as_current_span("bot.ping")
    async def ping(self, ctx: commands.Context) -> None:
        """Live dependency-health dashboard. The rendering and the live-edit loop
        live in src/ping.py; this is only the command surface. Reached only by a
        top-level -ping — the internal join/play path uses send_latency_line."""
        try:
            await run_health_dashboard(
                ctx,
                bot_latency=self.bot.latency,
                redis=self.redis,
                spotify=self.spotify,
                # The startup validation outcome, not just "is a client
                # configured": lets the Spotify row say *why* the source is
                # unusable without spending a doomed API call (see probe_spotify).
                spotify_status=self._spotify_status,
                archive=self.history_archive,
                debug_suffix=self._debug_suffix(ctx),
            )
        except Exception as e:
            await self._command_error(ctx, e)

    # ── Debug mode ────────────────────────────────────────────────────────────
    # The state machine is DebugSettings (src/debug.py); what stays here is the
    # command surface and the permission policy around the toggle.

    def _debug_suffix(
        self, ctx: commands.Context, *, host_metrics: bool = True
    ) -> Optional[str]:
        """The dashboards' debug footer, for a ctx rather than a guild."""
        return self.debug_settings.footer(ctx.guild, host_metrics=host_metrics)

    async def _hydrate_debug(self) -> None:
        """Feed DebugSettings.hydrate this bot's redis handle and guild list. Both
        cog_load and on_ready spawn it: bot.guilds is empty until READY, so
        cog_load's pass covers an extension reload and on_ready's a cold start."""
        await self.debug_settings.hydrate(self.redis, self.bot.guilds)

    @commands.Cog.listener()
    async def on_guild_remove(self, guild: discord.Guild) -> None:
        """Registration only — see DebugSettings.forget."""
        await self.debug_settings.forget(self.redis, guild.id)

    @commands.command(
        name="debug",
        aliases=["dbg"],
        brief="diagnostic snapshot; toggle debug mode",
        usage="[--enable | --disable]",
        help=(
            "Shows what this bot is running: versions, and Discord/voice state for "
            "this server. For the bot owner it also fills in host details — build, "
            "configuration, uptime, storage and health checks.\n\n"
            "`--enable` turns debug mode on for this server, which adds a footer to "
            "every embed the bot sends here — including the live Now Playing card, "
            "which refreshes its numbers alongside the progress bar. A reply's "
            "footer carries the trace id: paste it to the operator and they can find "
            "the exact request in the logs. (The Now Playing card shows the runtime "
            "numbers but no trace id — it is re-rendered under a different request "
            "every few seconds, so any one id there would be misleading.) `--disable` "
            "turns it back off. The choice is saved for this server and survives "
            "restarts; a server that has never set it follows the host's default. "
            "Toggling needs the **Manage Server** permission.\n\n"
            'Where `-ping` answers "are my dependencies up, and how fast?", this '
            'answers "what is running, and is it configured the way it should be?".'
        ),
        extras={
            "category": "Utility",
            # Read by cog_before_invoke to skip get_mp(): this command reports on a
            # guild's player, so manufacturing one to look at would be the observer
            # changing what it observes.
            "observation_only": True,
            "examples": ["-debug", "-debug --enable", "-debug --disable"],
            "note": (
                "Debug mode is per server and only changes what is DISPLAYED — "
                "never how the bot plays, queues or stores anything. Credentials "
                "are never shown, only whether they are set."
            ),
        },
    )
    # No validate_commands: diagnosing a bot needs no voice channel (-ping's
    # precedent). One in flight per guild, wait=False, since the snapshot does real
    # IO; cog_command_error renders the refusal.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @_tracer.start_as_current_span("bot.debug")
    async def debug(self, ctx: commands.Context, *, arg: str = "") -> None:
        try:
            action = debug_mode.parse_debug_arg(arg)
            if action is None:
                await ctx.send(
                    embed=notice_embed(
                        debug_mode.unknown_arg_message(arg), discord.Color.red()
                    )
                )
                return
            if action is not debug_mode.DebugAction.STATUS:
                await self._toggle_debug_mode(ctx, action)
                return
            inputs = await self._debug_inputs(ctx)
            # No typing indicator and no ctx.send: the dashboard sends its own
            # skeleton immediately and edits it as blocks land, so the reply IS the
            # acknowledgement. It uses channel.send to stay off the Now Playing host,
            # which an edit loop must not own (src/dashboard.py).
            await debug_mode.run_debug_dashboard(ctx, inputs)
        except Exception as e:
            await self._command_error(ctx, e)

    async def _debug_inputs(self, ctx: commands.Context) -> "debug_mode.DebugInputs":
        """Everything the snapshot cannot reach on its own (src/debug.py importing
        MusicBot would be a cycle)."""
        guild_id = ctx.guild.id if ctx.guild else None
        archive_enabled = history_archive_enabled()
        operator = await self._is_owner(ctx)
        # Asked symmetrically — not only when the password IS the default — because a
        # row that renders for False and vanishes for True makes its own absence the
        # answer. Moot while the whole Checks block is owner-only, and it stays right
        # if that ever loosens.
        default_password = (
            (using_default_postgres_password() and archive_enabled)
            if operator
            else None
        )
        return debug_mode.DebugInputs(
            debug_enabled=self.debug_settings.enabled(guild_id),
            debug_overridden=self.debug_settings.has_override(guild_id),
            debug_persisted=self.debug_settings.is_persisted(guild_id),
            players=len(self.mps),
            player=self.mps.get(guild_id) if guild_id is not None else None,
            redis=self.redis,
            store=GuildRedisStore(self.redis, guild_id)
            if self.redis is not None and guild_id is not None
            else None,
            # Structural: PostgresHistoryArchive satisfies ArchiveStatsReader, and
            # a cog built without an archive (tests, disabled tier) passes None.
            archive=cast(
                Optional["debug_mode.ArchiveStatsReader"], self.history_archive
            ),
            archive_enabled=archive_enabled,
            prometheus_url=debug_prometheus_url(),
            operator=operator,
            default_password=default_password,
            # Gated on `operator`: the card withholds its Runtime block from a
            # non-owner and says so, so the footer must not print those figures.
            debug_suffix=self._debug_suffix(ctx, host_metrics=operator),
        )

    async def _is_owner(self, ctx: commands.Context) -> bool:
        """Is the caller the bot owner? Fails CLOSED.

        `is_owner()` falls through to an `application_info()` REST call when neither
        owner_id nor owner_ids is configured (MusicBotApp sets neither), and it RAISES
        rather than returning False — a diagnostic must not disclose the host just
        because Discord blinked. discord.py caches the answer onto the bot afterwards,
        so this is one round trip per process, not per command.
        """
        try:
            return await ctx.bot.is_owner(ctx.author)
        except Exception as e:  # noqa: BLE001 — an unreachable owner is not an owner
            log.warning(f"owner check failed, denying: {type(e).__name__}: {e}")
            return False

    async def _toggle_debug_mode(
        self, ctx: commands.Context, action: "debug_mode.DebugAction"
    ) -> None:
        """Apply `--enable`/`--disable` to the invoking guild and confirm it."""
        if ctx.guild is None:
            await ctx.send(
                embed=notice_embed(
                    "Debug mode is set per server, so it can't be toggled from a "
                    f"direct message. It is currently "
                    f"**{'on' if self.debug_settings.default else 'off'}** here, following "
                    "the host's `DEBUG_MODE` default.",
                    discord.Color.orange(),
                )
            )
            return
        # A moderator action: the toggle is guild-wide and every member sees the
        # result on every reply, so it is not the invoking user's to make alone.
        # Reading `-debug` stays open to everyone; only writing is gated.
        author = ctx.author
        may_toggle = (
            isinstance(author, discord.Member) and author.guild_permissions.manage_guild
        ) or await self._is_owner(ctx)
        if not may_toggle:
            await ctx.send(
                embed=notice_embed(
                    "Debug mode changes what **every** embed in this server looks "
                    "like, so switching it needs the Manage Server permission. Run "
                    "`-debug` on its own to see the current state.",
                    discord.Color.red(),
                )
            )
            return
        enabled = action is debug_mode.DebugAction.ENABLE
        persisted = await self.debug_settings.toggle(self.redis, ctx.guild.id, enabled)
        # Say which kind of change this was. A guild told "on" that quietly reverts
        # on the next restart reads as the bot ignoring them, so a degraded write is
        # named rather than rounded up to success.
        durability = (
            "The setting is saved for this server."
            if persisted
            else "⚠️ It could not be saved (Redis is unavailable), so it applies "
            "until the bot restarts."
        )
        # Names what enabling publishes, at the moment the choice is made: the
        # footer reports the whole process's load, and the Now Playing card carries
        # it passively to everyone who can read the channel while music plays.
        scope = (
            " While it is on, every embed here — including the live Now Playing "
            "card — shows the bot process's load to anyone who can read the channel."
            if enabled
            else ""
        )
        await ctx.send(
            embed=notice_embed(
                f"Debug mode is now **{'on' if enabled else 'off'}** for this "
                "server. Embeds "
                + ("carry" if enabled else "no longer carry")
                + " a debug footer; nothing about playback changes either way."
                + scope
                + " "
                + durability,
                discord.Color.blue(),
            )
        )

    # ── Restart recovery listeners ────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Fires on cold start or session loss (not on WebSocket resume).
        Spawns a recovery task per guild so we don't block the event loop."""
        if self.redis is None:
            return
        # bot.guilds is empty until READY, so cog_load's pass covers an extension
        # reload and this one covers a cold start. Both are needed.
        spawn_background(self._hydrate_debug(), self._restore_tasks)
        for guild in self.bot.guilds:
            spawn_background(restore_guild(self, guild), self._restore_tasks)

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Registration only — the alone-disconnect state machine is
        VoiceWatchdog (src/recovery.py)."""
        await self.voice_watchdog.on_voice_state_update(member, before, after)


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusicBot(bot))
