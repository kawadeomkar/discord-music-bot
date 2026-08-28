import asyncio
import contextlib
import time
from dataclasses import dataclass, replace
from itertools import islice
from typing import (
    Any,
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
from src.musicplayer import InterjectOutcome, MusicPlayer
from src.play_placement import (
    NEXT_FLAG,
    NOW_FLAG,
    PlaceResult,
    PlaceStalled,
    PlaceVerdict,
    Placement,
    PlayArgs,
    PlayMode,
    PlayRegistry,
    PlayRequest,
    ResolveMode,
    check_voice_permissions,
    join_succeeded,
    play_key,
    play_takes_the_queue,
    resolve_mode_for,
    split_play_args,
)
from src.redis_client import (
    HISTORY_CACHE_LIMIT,
    GuildRedisStore,
    cache_get,
    cache_set,
)
from src.guild_queue import (
    QueueItem,
    RemoveMode,
    RemoveOutcome,
    item_label,
    matches_origin,
)
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
    build_embed,
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


class HistoryFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    limit: int = 10


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
    """What a `--next` confirmation says about when the song plays: the song it
    waits behind, and — since `--next` does not interject a paused song — that
    playback is still paused."""
    current = mp.current_song
    if current is None:
        # A claim with no current_song is loop() between taking the prefetch
        # result and starting it: the insert lands behind that song.
        if mp.queue.claim_outstanding():
            return "Plays after the song starting now."
        return "Nothing is playing, so it starts now."
    note = f"Plays after **{current.title or 'the current song'}**."
    if isinstance(voice_client, discord.VoiceClient) and voice_client.is_paused():
        note += " Playback is paused — `-resume` to carry on."
    return note


def _with_queue_position(item: QueueItem, position: int) -> QueueItem:
    """Re-mint one item's `queue_position`. A QueueObject is stamped in place, a
    frozen YTSource returns a copy — use the return value either way."""
    analytics = replace(item.analytics, queue_position=position)
    if isinstance(item, QueueObject):
        item.analytics = analytics
        return item
    return replace(item, analytics=analytics)


def _collection_note(
    url: str, queued: int, *, returns: str = "", head_playing: bool
) -> str:
    """What a `-play` that queued a whole collection tells the user: how many
    tracks landed, when the interrupted song returns, and the `-remove` undo.
    `head_playing` changes the undo: a playing song has no queue object for
    -remove to reach, so it names -skip."""
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
    """Ask-time `queue_position` for a song going to the FRONT: it waits behind the
    playing song and nothing else. An outstanding claim counts as that song even
    while current_song is None (loop() between taking the prefetch result and
    starting it). ±1 like enqueue_depth(): two `--next` in a row both record 1."""
    return 1 if mp.current_song is not None or mp.queue.claim_outstanding() else 0


def _rebase_positions(
    tracks: Sequence[QueueItem], minted_from: int, base: int
) -> Sequence[QueueItem]:
    """Move a resolved collection's `queue_position`s from one head depth to
    another; a no-op when the head has not moved, so the O(N) copy (milliseconds at
    5,000 tracks) runs only when another request placed in between, not under the
    lock on every enqueue."""
    if base == minted_from:
        return tracks
    return [_with_queue_position(t, base + offset) for offset, t in enumerate(tracks)]


def _head_depth(mp: MusicPlayer, placement: Placement) -> int:
    """`queue_position` for the first song an insert adds, read at the insert:
    the slot it actually takes. A cold start plays ahead of everything."""
    if placement is Placement.COLD_FRONT:
        return 0
    if placement is Placement.NEXT:
        return _front_insert_depth(mp)
    return mp.enqueue_depth()


async def _reply(
    ctx: commands.Context, embeds: Sequence[discord.Embed], reaction: str = "👍"
) -> None:
    """Confirm a placement that already happened. return_exceptions: the song is IN
    the queue, and a missing Add Reactions permission or a deleted invoking message
    must not render "Failed to queue song" over a song that plays."""
    results = await asyncio.gather(
        ctx.message.add_reaction(reaction),
        *(ctx.send(embed=embed) for embed in embeds),
        return_exceptions=True,
    )
    for failed in (r for r in results if isinstance(r, BaseException)):
        log.warning(f"Confirmation leg failed after the song was queued: {failed!r}")


def _join_failed_notice(query: str) -> discord.Embed:
    """A waiting request's own report of a failed cold-start join: join() reports
    only into the context of the request that created it."""
    return notice_embed(
        f"Couldn't join the voice channel, so your song wasn't queued: {_echo(query)}",
        discord.Color.red(),
    )


def _restore_unreachable_notice() -> discord.Embed:
    """The cold-start restore read never landed, so nothing was inserted."""
    return notice_embed(
        "Couldn't reach this server's saved queue, so your song wasn't queued — "
        "try again in a moment.",
        discord.Color.red(),
    )


def _place_stalled_notice(*, before_the_put: bool) -> discord.Embed:
    """What a stalled placement may claim. Waiting for the lock nothing was written
    and the song is honestly absent; inside the put the deque is appended before the
    mirror write, so the song may be queued and a "try again" would duplicate it."""
    if before_the_put:
        return notice_embed(
            "This server's queue is busy right now, so your song wasn't queued — "
            "try again in a moment.",
            discord.Color.red(),
        )
    return notice_embed(
        "This server's queue is busy right now, so your song may not have been "
        "queued — check `-queue` before trying again.",
        discord.Color.red(),
    )


def _cleared_title(cleared: int, dropped: int) -> str:
    """`-clear`'s heading, counting both halves: with nothing queued, the dropped
    requests ARE the work the command did."""
    songs = f"{cleared} {pluralize(cleared, 'song')} removed"
    if not dropped:
        return f"Queue cleared — {songs}"
    requests = f"{dropped} play {pluralize(dropped, 'request')} dropped"
    return f"Cleared — {songs}, {requests}" if cleared else f"Cleared — {requests}"


def _dropped_request_field(
    dropped: list[PlayRequest],
) -> Optional[list[tuple[str, str, bool]]]:
    """The resolving play requests a command dropped, as one embed field. None when
    it dropped none, which is what send_embed takes for no field at all."""
    if not dropped:
        return None
    return [
        (
            f"{len(dropped)} play {pluralize(len(dropped), 'request')} dropped",
            queue_message([safe_label(r.query, _ECHO_ROW_MAX) for r in dropped]),
            False,
        )
    ]


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
        # Per guild: the -play requests resolving, their place lock, and the
        # cold-start join they share. Created on demand, dropped once idle.
        self._plays = PlayRegistry()
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
        await self._plays.retire_player(guild.id, mp)
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
            # number > 1 is -play's in-flight cap; the renders keep their one slot.
            text = (
                f"Too many `{cmd}` requests are still resolving in this server — "
                "try again in a moment."
                if error.number > 1
                else f"A `{cmd}` request is already running in this server."
            )
            await ctx.send(embed=notice_embed(text, discord.Color.orange()))

    async def validate_commands(self, ctx: commands.Context) -> None:
        """before_invoke hook: rejects the command with a user-facing message
        if the author isn't in a usable voice channel."""
        vc = ctx.voice_client
        voice_client = vc if isinstance(vc, discord.VoiceClient) else None
        command_name = ctx.command.name if ctx.command is not None else ""
        msg = check_voice_permissions(
            ctx.author,
            voice_client,
            command_name,
            queue_control=play_takes_the_queue(ctx, voice_client),
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
        mode: ResolveMode,
    ) -> Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]:
        """Resolve a parsed URL/search source into something enqueueable: a
        ResolvedSpotifyPlaylist (titles still needing per-title YouTube resolution),
        a ResolvedYoutubePlaylist (already resolved), or a bare QueueObject.

        `analytics` is the command's ask-time head value, minted at dispatch;
        playlist tracks derive their per-track positions from it. `origin` is the
        raw command argument, carried onto every resulting item — for a collection
        the link, not the per-track search its expansion generated.

        `mode` is required and has no default: interjection resolves through this
        same helper, so "a search may go flat" cannot be decided from the input
        shape — only the caller knows whether the song must be playable on arrival."""
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            # Titles, not QueueObjects — _enqueue_playlist mints the YTSources
            # they become, carrying this command's analytics.
            titles = await self._require_spotify().playlist(source.id)
            if not titles:
                # Otherwise the enqueue below confirms "Queued playlist" with 👍
                # over nothing queued, which reads exactly like success.
                raise EmptyPlaylistError()
            return ResolvedSpotifyPlaylist(titles)
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
            # Only a search has a cheap metadata-only mode. A link's cost is the watch
            # page, which no yt-dlp option skips without failing YouTube's bot check.
            flat = mode is ResolveMode.FLAT_OK and (
                isinstance(source, SpotifySource)
                or (isinstance(source, YTSource) and source.ytsearch is not None)
            )
            return await YTDL.yt_source(
                ctx.author,
                search,
                ts=ts,
                redis=self.redis,
                query_source=query_source_of(source),
                analytics=analytics,
                user_input=origin,
                flat=flat,
            )

    @_tracer.start_as_current_span("bot.warm_front_track")
    async def _warm_front_track(
        self, tracks: Sequence[QueueItem], placement: Placement
    ) -> None:
        """Warm the stream URL of a playlist's head when it is about to play. Bulk
        enqueues pass prefetch=False (N extractions mint URLs that expire before
        playback reaches them), but under `--next` queue_put_next killed the loop's
        one-ahead prefetch, so the head would pay a full in-band extraction. A lazy
        Spotify entry has no URL yet; it resolves at dequeue."""
        if placement is not Placement.NEXT or not tracks:
            return
        head = tracks[0]
        if isinstance(head, QueueObject):
            await YTDL.prefetch_stream(head, redis=self.redis)

    @_tracer.start_as_current_span("bot.enqueue_playlist")
    async def _enqueue_playlist(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        qobj: Union[ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist],
        mp: MusicPlayer,
        req: PlayRequest,
        *,
        analytics: Analytics,
        origin: str,
        placement: Placement = Placement.TAIL,
    ) -> None:
        """Queue a resolved playlist under the place lock and notify the channel.
        Spotify playlists arrive as titles needing YouTube search, YouTube playlists
        pre-resolved. Positions are minted at the insert: `analytics` carries the
        ask time and its depth is replaced by the one the head takes."""
        # A playlist front-inserts in full, in order, under either flag. NEXT uses
        # queue_put_next: the loop's prefetch holds a claim a plain front-insert
        # lands behind. COLD_FRONT has no prefetch — the gate is shut.
        enqueue = {
            Placement.TAIL: mp.queue_put,
            Placement.COLD_FRONT: mp.queue_put_front,
            Placement.NEXT: mp.queue_put_next,
        }[placement]
        warning = timestamp_warning(source)
        warning_line = f"\n\n{warning}" if warning else ""
        # "Queued playlist" on its own reads as "at the back".
        next_suffix = " — plays next" if placement is Placement.NEXT else ""
        tracks: Sequence[QueueItem]
        if isinstance(qobj, ResolvedSpotifyPlaylist):
            titles = qobj.titles
            shown_titles = queue_message([safe_label(t, _ECHO_ROW_MAX) for t in titles])
            embed = build_embed(
                "Queued playlist" + next_suffix,
                f"Requested by: [{ctx.author.mention}]\n\n{shown_titles}{warning_line}",
                discord.Color.blue(),
            )
            # Built outside the lock: one YTSource per title, and the depth it
            # is minted against is almost always still the depth at the insert.
            provisional = _head_depth(mp, placement)
            tracks = spotify_playlist_to_ytsearch(
                titles,
                analytics=replace(analytics, queue_position=provisional),
                origin=origin,
            )
            log.info(f"spotify playlist track count: {len(tracks)}")
            async with self._plays.place(req) as verdict:
                if verdict.placed:
                    tracks = _rebase_positions(
                        tracks, provisional, _head_depth(mp, placement)
                    )
                    await enqueue(tracks, prefetch=False)
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
            embed = build_embed(
                f"Queued playlist — {count} {pluralize(count, 'song')}{next_suffix}",
                f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n"
                f"{skipped_line}\n{shown_titles}{warning_line}",
                discord.Color.blue(),
            )
            # Minted before the lock: at 5,000 tracks the pass is milliseconds of
            # event-loop time every sibling -play would wait out (_rebase_positions).
            provisional = _head_depth(mp, placement)
            tracks = _rebase_positions(tracks, 0, provisional)
            async with self._plays.place(req) as verdict:
                if verdict.placed:
                    tracks = _rebase_positions(
                        tracks, provisional, _head_depth(mp, placement)
                    )
                    await enqueue(tracks, prefetch=False)
        if not verdict.placed:
            await self._report_dropped(req, verdict)
            return
        await asyncio.gather(
            _reply(ctx, [embed]), self._warm_front_track(tracks, placement)
        )

    @staticmethod
    def _playing_next_embed(
        ctx: commands.Context, qobj: QueueObject, *, note: str
    ) -> discord.Embed:
        """The "Playing next" confirmation for `-play --next` and for an interjection
        whose song ended first; `note` says why the song is next. With nothing
        playing the title reads "Playing now", so it agrees with the note."""
        starts_now = note.startswith("Nothing is playing")
        lead = "▶️ Playing now" if starts_now else "▶️ Playing next"
        return build_embed(
            truncate_embed_title(f"{lead}: {qobj.title}"),
            f"Requested by: [{ctx.author.mention}]\n{note}",
            discord.Color.blue(),
            thumbnail=qobj.thumbnail,
        )

    @_tracer.start_as_current_span("bot.enqueue_single")
    async def _enqueue_single(
        self,
        ctx: commands.Context,
        qobj: QueueObject,
        mp: MusicPlayer,
        req: PlayRequest,
        *,
        placement: Placement = Placement.TAIL,
        note: str = "",
        warning: Optional[str] = None,
        follow_on: Sequence[QueueItem] = (),
    ) -> None:
        """Insert one resolved song under the place lock, then confirm. Under the
        lock: the put and the `queue_position` minted on it. The embeds are built
        and sent off the lock (the tail confirmation after the put, so it names the
        slot taken and re-hosts the live block when that slot is the head); see
        docs/ARCHITECTURE.md#play-placement. `warning` rides the confirmation or
        gets its own message; `follow_on` is a collection's tail behind the head."""
        vc = ctx.voice_client
        embeds: list[discord.Embed] = []
        # Carried to the end unless a branch folds it into its own embed instead.
        warning_embed = (
            notice_embed(warning, discord.Color.orange())
            if warning is not None
            else None
        )
        should_show_queued = False
        if placement is Placement.COLD_FRONT:
            # A restored queue is non-empty but sits BEHIND this song, so the resume
            # notice replaces the confirmation; built while the queue holds only the
            # restored entries.
            resume_notice = mp.build_resume_notice_embed(qobj)
            if resume_notice is not None:
                embeds.append(resume_notice)
        elif placement is Placement.NEXT:
            # No "Est. playing at": the ETA walk seeds from the current song's
            # FULL duration as a proxy for what is left of it, which is badly
            # wrong for the very next slot. It names the song it waits behind.
            embeds.append(
                self._playing_next_embed(ctx, qobj, note=_plays_after_note(mp, vc))
            )
        else:
            # A note is the only word the user gets about tracks queued behind
            # this one, so an empty queue does not suppress the field.
            should_show_queued = (
                bool(note)
                or mp.queue.qsize() > 0
                or (isinstance(vc, discord.VoiceClient) and vc.is_playing())
            )
            if should_show_queued:
                warning_embed = None  # it rides the confirmation, built below
            # Otherwise the song starts now and the NP card speaks for it, so the
            # warning needs its own message.
        if warning_embed is not None:
            embeds.append(warning_embed)
        async with self._plays.place(req) as verdict:
            if verdict.placed:
                depth = _head_depth(mp, placement)
                qobj.analytics = replace(qobj.analytics, queue_position=depth)
                if placement is Placement.COLD_FRONT:
                    await mp.queue_put_front(qobj)
                elif placement is Placement.NEXT:
                    await mp.queue_put_next(qobj)
                else:
                    await mp.queue_put(qobj)
                    if follow_on:
                        # Behind the head, in its order, re-minted from the head's
                        # depth: play_history keeps whatever number is on them.
                        await mp.queue_put(
                            [
                                _with_queue_position(item, depth + offset)
                                for offset, item in enumerate(follow_on, start=1)
                            ],
                            prefetch=False,
                        )
                log.info(f"play ({placement.value}) qsize: {mp.queue.qsize()}")
        if not verdict.placed:
            await self._report_dropped(req, verdict)
            return
        if should_show_queued:
            # After the put, off the lock, so the card carries the slot the song
            # took. When this song IS the head, the block's "Up next" card is the
            # same card; re-host the live block instead — dedicated, since a
            # response host with no own embeds strip-edits to a blank message.
            if mp.queue.peek_next() is qobj and await mp.repin_now_playing():
                # What the card would have carried: the note and the warning.
                said = "\n\n".join(text for text in (note, warning) if text)
                if said:
                    color = discord.Color.orange() if warning else discord.Color.blue()
                    embeds.append(notice_embed(said, color))
            else:
                embeds.append(
                    mp.build_queued_song_embed(qobj, note=note, warning=warning)
                )
        await _reply(ctx, embeds)

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
                "long playlist takes a few seconds to finish queueing. Requests sent "
                "meanwhile are looked up alongside it and land as each one is ready — "
                "a `--now` sent behind a long playlist interrupts as soon as its own "
                "song resolves. `-clear` and `-stop` also drop requests still being "
                "looked up."
            ),
        },
    )
    # Requests resolve concurrently and serialize only at _place(). The cap is
    # raised here, before _play's except, so it reaches cog_command_error rather
    # than rendering as "Failed to queue song".
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.play")
    async def play(self, ctx: commands.Context, *, url: str) -> None:
        # Consume-rest, so a multi-word search arrives whole — it is what -remove
        # matches on. The strip covers callers that bypass discord.py's parser.
        args = split_play_args(url.strip())
        # Bound here, not after the resolve: every failure path hands this exact
        # player to _abandon_cold_start, and a get_mp() after its cleanup() would
        # start a fresh one. cog_before_invoke already created it.
        req = self._plays.register(
            ctx, query=args.query, mp=self.get_mp(ctx), mode=args.mode
        )
        try:
            await self._play(ctx, args, req)
        finally:
            self._plays.retire(req)

    async def _report_dropped(self, req: PlayRequest, verdict: PlaceResult) -> None:
        """Tell the author why a resolved request did not place. An ordinary
        command reply, so it carries the NP block like any other."""
        if verdict.verdict is PlaceVerdict.VOICE:
            text = verdict.refusal
        elif req.dropped_by:
            text = (
                f"Your play request was dropped — `-{req.dropped_by}` ran while "
                "it was resolving."
            )
        elif verdict.verdict is PlaceVerdict.CLEARED:
            text = "Your play request was dropped — the queue was cleared while it was resolving."
        else:
            text = (
                "Your play request was dropped — the session ended before your "
                "song could be queued."
            )
        # Which one: a user with three resolving requests gets three of these, and
        # the dropping command's own listing is not reachable from every path.
        text = f"{text}\n{_echo(req.query)}"
        await req.ctx.send(embed=notice_embed(text, discord.Color.red()))

    @staticmethod
    async def _report_stalled(req: PlayRequest, stall: PlaceStalled) -> None:
        await req.ctx.send(
            embed=_place_stalled_notice(before_the_put=stall.before_the_put)
        )

    async def _play(
        self, ctx: commands.Context, args: PlayArgs, req: PlayRequest
    ) -> None:
        """The body behind -play, taking the argument already split and the
        request the registry admitted."""
        trace.get_current_span().set_attribute("play.mode", args.mode.value)
        # ONE rebind, so every `origin=url` below is the query with the flag off:
        # a leaked flag persists a user_input -remove cannot match. read_rest hands
        # quotes through, and a quoted origin is one -remove would match literally.
        url = unquote_argument(args.query)
        # The error title names the branch, not the flag: `-p --now x` on an idle
        # bot queues like any other -play.
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

                # Bound before the join: every failure path hands this exact player
                # to _abandon_cold_start, and a get_mp() after its cleanup() would
                # start a fresh one. cog_before_invoke already created it.
                mp = self.get_mp(ctx)
                vc = ctx.voice_client
                # A narrowed Optional: VoiceProtocol carries neither is_playing nor
                # is_paused, so a bool would leave vc unnarrowed at both use sites.
                live_vc = (
                    vc
                    if isinstance(vc, discord.VoiceClient)
                    and mp.current_song is not None
                    and (vc.is_playing() or vc.is_paused())
                    else None
                )
                # Decided BEFORE the resolve, which is where whether a song is live
                # can change.
                if live_vc is not None:
                    if args.mode is PlayMode.NOW:
                        interjecting = True
                        return await self._interject_flow(ctx, url, mp, live_vc, req)
                    if live_vc.is_paused() and args.mode is not PlayMode.NEXT:
                        # Paused → interject: appending would bury the request behind
                        # a paused song. The interrupted song returns PLAYING. `--next`
                        # is excluded — it is next either way.
                        interjecting = True
                        return await self._interject_flow(
                            ctx,
                            url,
                            mp,
                            live_vc,
                            req,
                            resume_paused=False,
                            require_paused=True,
                        )

                source = parse_input(url)

                notice = await self._resolve_and_place(ctx, args, req, mp, source, url)
                if notice is not None:
                    await ctx.send(embed=notice)

            except PlaceStalled as stall:
                # The interject route: no gate hold to unwind, nothing to abandon.
                await self._report_stalled(req, stall)
            except Exception as e:
                await self._command_error(
                    ctx,
                    e,
                    title="Failed to play song now"
                    if interjecting
                    else "Failed to queue song",
                )

    async def _resolve_and_place(
        self,
        ctx: commands.Context,
        args: PlayArgs,
        req: PlayRequest,
        mp: MusicPlayer,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        url: str,
    ) -> Optional[discord.Embed]:
        """Resolve, then insert under the place lock. Returns the notice for a
        request that did not insert, or None. Returned rather than sent: on the cold
        path a teardown decision (_abandon_cold_start reads the hold count) must not
        be followed by an await before the gate hold is released."""
        qobj: Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]
        async with contextlib.AsyncExitStack() as stack:
            # Not connected: this song jumps ahead of any queue restored from Redis.
            # The flag decides the analytics shortcut and the join dance; the insert
            # position is `placement`. A running join counts as cold — discord.py
            # registers the client BEFORE the handshake completes (join_succeeded).
            in_flight_join = self._plays.join_in_flight(req.guild_id)
            cold_start = not ctx.voice_client or in_flight_join
            if cold_start:
                placement = Placement.COLD_FRONT
            elif args.mode is not PlayMode.NORMAL:
                # Both flags: `--now` reaches here only with nothing to interrupt,
                # and that state lasts the length of every resolve.
                placement = Placement.NEXT
            else:
                placement = Placement.TAIL
            # The message's snowflake time, so the wait covers gateway delivery.
            # The depth is minted at the insert, under the place lock.
            analytics = Analytics(
                queued_at=ctx.message.created_at.timestamp(), queue_position=0
            )
            resolve_started = time.monotonic()
            if cold_start:
                # Hold the gate across the join, which opens it the moment the
                # handshake lands — before queue_source has finished extracting.
                # Released on exiting the stack, after the front insertion.
                await stack.enter_async_context(mp.defer_playback())
                # One join per guild, concurrent with this resolve: voice
                # handshake and yt-dlp extraction have no data dependency.
                join, owns_join = self._plays.cold_join(
                    req,
                    joiner=lambda: ctx.invoke(self.join),
                    tracked=self._restore_tasks,
                )
                try:
                    async with self._plays.resolve_slot(req):
                        qobj = await self.queue_source(
                            ctx,
                            source,
                            analytics=analytics,
                            origin=url,
                            mode=resolve_mode_for(placement),
                        )
                except BaseException:
                    # Alone on this cold start (the hold count is this command's),
                    # cancel the join before the teardown, or join() rebuilds the
                    # player it then finds missing. Another holder owns the join.
                    if mp.playback_holds == 1 and not join.done():
                        join.cancel()
                        with contextlib.suppress(asyncio.CancelledError, Exception):
                            await join
                    # Full cleanup: cog_before_invoke already started a loop() that
                    # would otherwise zombie for 300s on queue.get() and leave
                    # clear_connection() unfired — spurious crash recovery.
                    await self._abandon_cold_start(ctx, mp)
                    raise
                with contextlib.suppress(Exception):
                    # The join task runs on the CREATOR's context, so a waiter's
                    # trace has to say the join was somebody else's.
                    if not owns_join:
                        trace.get_current_span().set_attribute("play.join_shared", True)
                    await asyncio.shield(join)
                # Inserting onto a join that produced no usable client hands the
                # loop a song it can only raise on.
                if not join_succeeded(ctx):
                    # join() swallows its own error, so this span and this line are
                    # the only record of the drop. Synchronous through the return.
                    trace.get_current_span().set_attribute(
                        "play.dropped_by", "join_failed"
                    )
                    log.warning(f"Cold-start join left no voice client: {url}")
                    await self._abandon_cold_start(ctx, mp)
                    # The creator was told by join() itself. Returned, not sent: no
                    # await may land between the teardown decision and the release.
                    return None if owns_join else _join_failed_notice(url)
            else:
                async with self._plays.resolve_slot(req):
                    qobj = await self.queue_source(
                        ctx,
                        source,
                        analytics=analytics,
                        origin=url,
                        mode=resolve_mode_for(placement),
                    )
            trace.get_current_span().set_attribute(
                "play.resolve_secs", round(time.monotonic() - resolve_started, 3)
            )

            log.info(f"Voice client: {ctx.voice_client}")

            # Every placement waits: put_front LPUSHes the list restore_entries
            # replays in memory, so inserting first double-queues this song, and a
            # put() before the replay lands this song ahead of entries Redis lists
            # behind it. Bounded, since the pool sets no socket_timeout.
            if not await mp.wait_for_restore(timeout=RESTORE_WAIT_SECS):
                # Cold start ONLY: _abandon_cold_start cancels the player's tasks
                # and disconnects it, which on a warm player would stop the music
                # over a Redis blink.
                if cold_start:
                    await self._abandon_cold_start(ctx, mp)
                    return _restore_unreachable_notice()
                return notice_embed(
                    "Still loading this server's saved queue — try again in a moment.",
                    discord.Color.orange(),
                )

            try:
                if isinstance(qobj, QueueObject):
                    await self._enqueue_single(
                        ctx,
                        qobj,
                        mp,
                        req,
                        placement=placement,
                        warning=timestamp_warning(source),
                    )
                else:
                    await self._enqueue_playlist(
                        ctx,
                        source,
                        qobj,
                        mp,
                        req,
                        placement=placement,
                        analytics=analytics,
                        origin=url,
                    )
            except PlaceStalled as stall:
                if cold_start and not self._cold_start_left_something_playable(ctx, mp):
                    await self._abandon_cold_start(ctx, mp)
                return _place_stalled_notice(before_the_put=stall.before_the_put)
            if cold_start and not req.placed:
                # Refused at the lock after the join put the bot in the channel:
                # tear down like the other exits, or it sits in an empty channel
                # until the 300s idle — unless the channel has music to play
                # (_cold_start_left_something_playable).
                if not self._cold_start_left_something_playable(ctx, mp):
                    await self._abandon_cold_start(ctx, mp)
                return None
        return None

    async def _resolve_interjection_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        *,
        origin: str,
    ) -> tuple[QueueObject, list[QueueItem]]:
        """Resolve an interjection's input into (head, everything behind it). The
        head must be a resolved QueueObject to interrupt with; the tail may hold
        lazy YTSources. The interrupted song returns after the whole playlist, and
        one `-remove <the link>` takes it all back out. `origin` is the raw command
        argument — for a playlist the link, not the generated titles."""
        # Ask-time analytics: the message's snowflake time, depth 0 for the head.
        # Tracks behind it derive 1, 2, … from this base; the caller re-mints the
        # head's depth on the two paths where it queues instead.
        analytics = Analytics(
            queued_at=ctx.message.created_at.timestamp(), queue_position=0
        )
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            titles = await self._require_spotify().playlist(source.id)
            if not titles:
                # Same class the YouTube path raises: "ValueError: Playlist has no
                # tracks" reaches the user verbatim through _command_error.
                raise EmptyPlaylistError()
            yts = spotify_playlist_to_ytsearch(
                titles, analytics=analytics, origin=origin
            )
            # Only the head is resolved — it has to be playable to interrupt with,
            # which is also why this takes yt_source's full path rather than the flat
            # one. The rest stay lazy searches resolved at dequeue, so a 100-track
            # album does not pay 100 searches up front.
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
        # FULL, not the placement default: interject() stops the current song, so this
        # head has to be playable before anything is stopped.
        qobj = await self.queue_source(
            ctx, source, analytics=analytics, origin=origin, mode=ResolveMode.FULL
        )
        assert isinstance(qobj, QueueObject)
        return qobj, []

    @_tracer.start_as_current_span("bot.interject_flow")
    async def _interject_flow(
        self,
        ctx: commands.Context,
        url: str,
        mp: MusicPlayer,
        vc: discord.VoiceClient,
        req: PlayRequest,
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
        async with self._plays.resolve_slot(req):
            qobj, follow_on = await self._resolve_interjection_source(
                ctx, source, origin=url
            )
        # The head only: `interjected` is attribution, which song cut the line.
        qobj.interjected = True

        # Warm the stream-URL cache before interrupting: a cache miss at dequeue puts
        # seconds of yt-dlp dead air between the interrupt and the new song. Awaited,
        # not spawned — the current song plays through the wait. No-op without Redis;
        # also back-fills duration/thumbnail for the embeds below.
        #
        # The head only: warming N tracks would be N concurrent extractions
        # minting URLs that expire before playback reaches them.
        await YTDL.prefetch_stream(qobj, redis=self.redis)

        # Before the lock: the neutralize can wait on a prefetch pinned in the
        # yt-dlp executor, which under _place would hold the guild's lock.
        await mp.settle_prefetch()

        outcome: Optional[InterjectOutcome] = None
        resumed = False
        async with self._plays.place(req) as verdict:
            if not verdict.placed:
                pass
            elif require_paused and not vc.is_paused():
                # Resumed during the resolve, so the reason to interject is gone:
                # append instead. The append takes the lock again on its own.
                resumed = True
            else:
                outcome = await mp.interject(
                    qobj, vc, resume_paused=resume_paused, follow_on=follow_on
                )
                if outcome is None:
                    # The song ended during the resolve. Insert qobj directly, at
                    # the front — the user asked for "now" and this window can be
                    # seconds long. It interrupted nothing, so the marker comes off.
                    qobj.interjected = False
                    # interject() also returns None when the loop moved on to a
                    # DIFFERENT song, which this insert waits behind: depth 1.
                    qobj.analytics = replace(
                        qobj.analytics, queue_position=_front_insert_depth(mp)
                    )
                    # queue_put_next: the embed promises "next", and the loop's
                    # prefetch holds a claim a bare front-insert would land behind.
                    # prefetch=False — the stream URL was warmed above.
                    await mp.queue_put_next([qobj, *follow_on], prefetch=False)
        if not verdict.placed:
            await self._report_dropped(req, verdict)
            return

        if resumed:
            # Clear the marker: a queued song must not trigger replace semantics
            # later. The interjection's 0 is replaced at the insert.
            qobj.interjected = False
            note = (
                _collection_note(url, len(follow_on) + 1, head_playing=False)
                if follow_on
                else ""
            )
            await self._enqueue_single(
                ctx,
                qobj,
                mp,
                req,
                note=note,
                warning=timestamp_warning(source),
                follow_on=follow_on,
            )
            return

        if outcome is None:
            note = (
                "The song being interrupted already ended — "
                "queued to play next instead."
            )
            if follow_on:
                # Nothing was interrupted, so the head is QUEUED rather than
                # playing: it counts, and -remove reaches it.
                note += _collection_note(url, len(follow_on) + 1, head_playing=False)
            await asyncio.gather(
                ctx.send(embed=self._playing_next_embed(ctx, qobj, note=note)),
                ctx.message.add_reaction("⏯️"),
            )
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
            # The interrupted song waits behind the whole playlist, so the reply
            # says so and names the undo (`-remove <the link>` matches user_input).
            # Tail only, head_playing: the playing song has no queue object.
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
            "disconnects the bot from the voice channel. Play requests still "
            "being looked up are dropped and listed.\n\n"
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
            # After the teardown, with no await between, so no request places into
            # a player that no longer exists. Unconditional: a cold-start -play may
            # be resolving before there is a client to find.
            dropped = self._plays.inflight(play_key(ctx), "stop")
            if dropped:
                await send_embed(
                    ctx,
                    "Stopped",
                    "Play requests still resolving were dropped.",
                    discord.Color.red(),
                    fields=_dropped_request_field(dropped),
                )
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
                    joined = join_succeeded(ctx)
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

    @staticmethod
    def _cold_start_left_something_playable(
        ctx: commands.Context, mp: MusicPlayer
    ) -> bool:
        """Whether a cold start refused AFTER its join should leave the session up:
        with the connection up and the queue not empty, the songs are a sibling's
        or a restored queue's, and a teardown would disconnect a working session.
        Synchronous, so the caller decides before its gate release."""
        return join_succeeded(ctx) and mp.queue.display_size() > 0

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
            "Randomly reorders the songs waiting in the queue. Needs at least 4 "
            "queued songs to have any effect. The song currently playing is left "
            "alone — shuffling only touches what comes after it."
        ),
        extras={"category": "Queue", "examples": ["-shuffle"]},
    )
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
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
            "Removes every song waiting in the queue and lists what was dropped, "
            "along with any play requests still being looked up. The song "
            "currently playing keeps going — use `-skip` to move past it "
            "or `-stop` to end the session entirely."
        ),
        extras={"category": "Queue", "examples": ["-clear", "-c"]},
    )
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
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
            # Right after the clear, with no await between: every request still
            # resolving fails its generation check at the insert and reports
            # itself; this names them here too.
            dropped = self._plays.inflight(play_key(ctx), "clear")
            if not cleared and not dropped:
                await ctx.send(
                    embed=notice_embed(
                        "The queue is already empty.", discord.Color.orange()
                    )
                )
                return
            description = (
                queue_message([safe_label(t, _ECHO_ROW_MAX) for t in cleared])
                if cleared
                else "The queue was already empty."
            )
            await asyncio.gather(
                ctx.message.add_reaction("🗑️"),
                send_embed(
                    ctx,
                    _cleared_title(len(cleared), len(dropped)),
                    description,
                    discord.Color.red(),
                    fields=_dropped_request_field(dropped),
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
            "match a song queued from a full `youtube.com` one.\n\n"
            "Play requests still being looked up match the same way and are "
            "dropped before they can be queued, so a link you have only just "
            "typed can be taken back during the wait."
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
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
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
            # A -remove during a resolve is routine (a resolve can run 99s). Matched
            # the way the queue matches — a link literally, text case-folded — so
            # the same argument takes out the pending request and the queued song.
            dropped = self._plays.inflight(
                play_key(ctx), "remove", lambda r: matches_origin(needle, r.query)
            )
            outcome = await mp.queue_remove(needle)
            positions = outcome.positions
            if not positions:
                await send_embed(
                    ctx,
                    "",
                    (
                        f"No queued songs found matching: {_echo(needle)}"
                        if not dropped
                        else f"Nothing queued matched {_echo(needle)} yet — "
                        "it was still being looked up."
                    ),
                    discord.Color.red() if not dropped else discord.Color.orange(),
                    fields=_dropped_request_field(dropped),
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
