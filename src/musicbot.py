import asyncio
import contextlib
from dataclasses import dataclass
from itertools import islice
from typing import (
    Any,
    Optional,
    Union,
    assert_never,
)
from collections.abc import AsyncGenerator, Coroutine

import discord
from discord.ext import commands

import redis.asyncio as aioredis

from src.config import (
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
    spotify_enabled,
)
from src.musicplayer import MusicPlayer
from src.redis_client import HISTORY_CACHE_LIMIT, GuildRedisStore
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    spotify_titles_to_ytsearch,
)
from src.spotify import Spotify, SpotifyAuthError, SpotifyCollection, TrackPage
from src.youtube import YTDL, ExtractionError, QueueObject
from contextvars import Token

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

from src.ping import run_health_dashboard, send_latency_line
from src.telemetry import get_tracer
from src.util import (
    cancel_task,
    fmt_duration,
    history_embeds,
    notice_embed,
    pluralize,
    queue_message,
    record_span_error,
    send_embed,
    spawn_background,
    trace_footer,
    truncate_embed_title,
    get_logger,
)

log = get_logger(__name__)
_tracer = get_tracer(__name__)


class SpotifyDisabledError(Exception):
    """Raised when a Spotify link is played but Spotify support isn't usable.

    Carries the SpotifyStatus so the message distinguishes the two failure modes:
    no credentials were configured (DISABLED) vs. credentials were configured but
    rejected by Spotify at startup (INVALID). The message is user-facing:
    _command_error renders it (with the class name) into the error embed the
    requester sees, so it explains the config gap and points at the sources that
    still work.
    """

    def __init__(self, status: SpotifyStatus) -> None:
        self.status = status
        if status is SpotifyStatus.INVALID:
            message = (
                "Spotify links aren't available right now — this bot has Spotify "
                "credentials configured, but Spotify rejected them at startup "
                "(check SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET). "
                "Try a YouTube or SoundCloud link, or just search by name."
            )
        else:  # DISABLED (and any unexpected value — safest generic message)
            message = (
                "Spotify links aren't available on this bot — it was started without "
                "Spotify credentials (SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET). "
                "Try a YouTube or SoundCloud link, or just search by name."
            )
        super().__init__(message)


HISTORY_MIN_LIMIT = 1
# The ceiling is the display cache depth — -history reads the in-memory cache,
# or the Redis list when the cache is cold. See docs/HISTORY_OVERHAUL_PLAN.md §5.
HISTORY_MAX_LIMIT = HISTORY_CACHE_LIMIT
# 8 song embeds + the ≤2-embed NP block MusicContext.send may prepend = 10,
# Discord's per-message cap — so the block always fits and is never shed.
HISTORY_EMBEDS_PER_MESSAGE = 8


class HistoryFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    limit: int = 10


# Bounds on the per-guild collection-enqueue lock (§3.5.3 of the design doc).
# A FIFO that can be entered but not exited is worse than no FIFO; each bound
# closes a failure mode the naive version has.
#
# _ENQUEUE_WAIT_SECS MUST stay well under musicplayer._PLAYBACK_GATE_TIMEOUT
# (300s): a lock-waiter's player already exists — cog_before_invoke's get_mp()
# built and start()ed it before any command body ran — so on a disconnected
# bot the waiter's gate clock is already ticking while it waits. A wait bound
# at or above the gate timeout would let the player be torn down underneath a
# still-waiting command, which would then enqueue into a dead guild (the C2
# failure, reached through the corrected door). A test pins the inequality.
_ENQUEUE_WAIT_SECS = 60.0
# Beyond this many queued waiters, decline instead of stacking further — a
# user spamming -play with collections would otherwise pile up unbounded
# waiters that all fire into the queue at once when the drain finishes.
_ENQUEUE_MAX_WAITERS = 5


class _GuildEnqueueLock:
    """A guild's collection-enqueue slot: the lock plus its waiter count.

    asyncio.Lock specifically — it is documented fair ("thread always waits
    its turn"), so waiting collections land in arrival order. discord.py's
    max_concurrency semaphore was rejected for this job: per-Command buckets
    can't serialize play against shuffle, ctx.invoke bypasses prepare() (the
    -playnow→play delegation would escape it), and its acquire fast-path
    barges past just-woken waiters (design doc §3.5.2).
    """

    __slots__ = ("lock", "waiters")

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.waiters = 0


class ResolvedSpotifyStream:
    """A Spotify collection resolving page-by-page. Page 1 is ALREADY IN
    FLIGHT: queue_source starts it as a task, so the first HTTP round-trip
    runs concurrently with whatever the caller does next — for -play's
    front path that is the voice-join handshake, keeping the Spotify call
    inside the join overlap instead of serializing after it (design §3.1).

    Not a dataclass: it owns a live task and a generator and must be closed
    exactly once — aclose() or a full drain, never neither. Abandoning the
    generator instead defers finalization to asyncio's asyncgen hooks, which
    raise GeneratorExit at an arbitrary later point; under
    filterwarnings=["error"] the resulting warning is a hard test failure.
    """

    def __init__(self, kind: SpotifyType, pages: AsyncGenerator[TrackPage]) -> None:
        self.kind = kind
        self.pages = pages
        # __anext__() (not anext()) so the type checker sees a Coroutine, which
        # create_task requires. This is what starts the page-1 HTTP call NOW.
        self.first_page: asyncio.Task[TrackPage] = asyncio.create_task(
            pages.__anext__()
        )

    async def aclose(self) -> None:
        """Cancel an unconsumed first page and finalize the generator NOW."""
        if not self.first_page.done():
            self.first_page.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.first_page
        await self.pages.aclose()


@dataclass
class ResolvedYoutubePlaylist:
    """A YouTube playlist already resolved to playable QueueObjects.

    Carries its own playlist_url so the enqueue path never needs the original
    source object — the correlation "a ResolvedYoutubePlaylist always arrives
    with a YTSource" is now expressed by construction instead of a runtime
    assert (built in queue_source, where the source is already narrowed).
    """

    tracks: list[QueueObject]
    playlist_url: str


@dataclass
class _StreamDrain:
    """The state _begin_stream_enqueue hands to _drain_stream_tail: everything
    the tail needs to keep enqueueing pages after the playback gate opened."""

    resolved: ResolvedSpotifyStream
    generation: int  # snapshot the compare-and-put checks every page against
    enqueued: int
    total: int  # collection.total — an upper bound for playlists (§3.3)
    completion_notice: bool  # multi-page playlist: report the real count at the end


def _is_collection(
    source: Union[SpotifySource, YTSource, SoundcloudSource],
) -> bool:
    """True when this source enqueues more than one entry — the tier-1 rule.

    Knowable BEFORE any I/O (parse_input is pure string work), which is what
    lets play acquire the collection lock ahead of the AsyncExitStack without
    holding a playback-gate deferral while it waits. YT playlists count: their
    single batch put would otherwise land between a Spotify stream's pages and
    split the collection. Single tracks and searches never take the lock —
    they append at the current tail immediately, accepted interleaving
    (design §6: an immediate single can even land ahead of a collection that
    arrived earlier and is still waiting).
    """
    if isinstance(source, SpotifySource):
        return source.type in (SpotifyType.PLAYLIST, SpotifyType.ALBUM)
    return isinstance(source, YTSource) and source.type is YTType.PLAYLIST


def _check_voice_permissions(
    author: Union[discord.Member, discord.User],
    voice_client: Optional[discord.VoiceClient],
    command_name: str,
) -> Optional[str]:
    """Returns an error message string if validation fails, None if OK."""
    if isinstance(author, discord.User):
        return f"You must be a member of this channel {author}"
    if not author.voice or not author.voice.channel:
        return f"You are not connected to a voice channel, you silly baka {author}"
    if (
        command_name != "play"
        and voice_client is not None
        and voice_client.channel != author.voice.channel
    ):
        return f"Bot is already being used in channel {voice_client.channel}"
    return None


async def _typing_keepalive(ctx: commands.Context) -> None:
    try:
        async with ctx.typing():
            await asyncio.sleep(3600)  # held open until cancelled
    # Catch Exception only — NOT CancelledError. This task is cancelled by
    # background_typing() on the way out, and letting that CancelledError
    # propagate is what marks the task genuinely cancelled (task.cancelled()
    # True) instead of completing normally; swallowing it here would defeat
    # cooperative cancellation and stop a shutdown/outer-timeout cancellation
    # at this frame. Typing failures, on the other hand, are purely cosmetic
    # and must never surface — hence the Exception catch.
    except Exception:
        pass  # cosmetic — never let typing failures surface


@contextlib.asynccontextmanager
async def background_typing(ctx: commands.Context) -> AsyncGenerator[None]:
    """Non-blocking ctx.typing(): the first POST /typing runs in a background
    task so the command body starts immediately; the keepalive is held open until the
    body finishes, then cancelled. The whole CM lives inside the task — never enter/exit
    Typing manually across tasks (see docs/PERFORMANCE_PLAN.md §2.2)."""
    task = asyncio.create_task(_typing_keepalive(ctx))
    try:
        yield
    finally:
        task.cancel()


class MusicBot(commands.Cog):
    """
    class for music bot
    """

    __slots__ = (
        "bot",
        "mps",
        "spotify",
        "_spotify_status",
        "redis",
        "_active_spans",
        "_alone_timers",
        "_restore_tasks",
    )

    def __init__(self, bot: commands.Bot) -> None:
        self.bot = bot
        # HACK: getattr() hides MusicBot's real dependency on MusicBotApp.redis.
        # `bot` is typed commands.Bot, but `redis` is a MusicBotApp attribute, so the
        # access is spelled as getattr() to keep the type checker quiet. getattr()
        # returns Any, which downgrades the Optional[aioredis.Redis] annotation from a
        # checked type to an unchecked assertion: renaming MusicBotApp.redis silently
        # degrades every guild to no-Redis — no persistence, no crash recovery — instead
        # of failing at startup or at type-check time. Nothing here would go red.
        # Fix: type `bot` as "MusicBotApp" under `if TYPE_CHECKING` (src/main.py already
        # uses that idiom to break this exact import cycle), or declare a two-line
        # Protocol carrying `redis: Optional[aioredis.Redis]`.
        self.redis: Optional[aioredis.Redis] = getattr(bot, "redis", None)
        # Spotify is optional: only build the client when credentials are present.
        # When None, playing a Spotify link raises SpotifyDisabledError; every other
        # source (YouTube, SoundCloud, search) is unaffected. See _require_spotify.
        self.spotify: Optional[Spotify] = (
            Spotify(redis=self.redis) if spotify_enabled() else None
        )
        # Status starts ENABLED when credentials are present and DISABLED when
        # they're absent. cog_load() then probes the live API and downgrades to
        # INVALID if the credentials don't authenticate. The optimistic ENABLED
        # start is safe because cog_load runs inside setup_hook, before the
        # gateway connects — no command can arrive until the probe has resolved.
        self._spotify_status: SpotifyStatus = (
            SpotifyStatus.ENABLED
            if self.spotify is not None
            else SpotifyStatus.DISABLED
        )
        self.mps: dict[int, MusicPlayer] = {}
        # Per-guild collection-enqueue serialization (design §3.5.1, M5→b):
        # only collection enqueues and -shuffle take this — singles never wait.
        # Lives on the cog, NOT on MusicPlayer: cleanup() destroys players, and
        # a lock destroyed mid-wait orphans its waiters. Grows one small entry
        # per guild ever seen and is never pruned — an entry with live waiters
        # cannot be popped safely, and the footprint is ~100B per guild.
        self._enqueue_locks: dict[int, _GuildEnqueueLock] = {}
        # id(ctx) → (span, the token otel_context.attach() returns, which detach()
        # requires back — `object` does not satisfy it.
        self._active_spans: dict[int, tuple[Span, Token[Context]]] = {}
        self._alone_timers: dict[int, asyncio.Task] = {}
        self._restore_tasks: set[asyncio.Task] = set()

    async def cog_load(self) -> None:
        """Kick off Spotify credential validation without blocking startup.

        discord.py awaits this when the cog is added — inside setup_hook, before
        the bot connects — so anything awaited here delays the connection. The
        probe is a live network call, so it's spawned as a fire-and-forget
        background task instead: startup proceeds immediately and _spotify_status
        stays optimistically ENABLED until the probe resolves. With no credentials
        there's nothing to check."""
        if self.spotify is None:
            return
        spawn_background(self._validate_spotify_credentials(), self._restore_tasks)

    async def _validate_spotify_credentials(self) -> None:
        """Background credential probe (spawned by cog_load, never awaited on the
        startup path).

        Only an authentication rejection — SpotifyAuthError — marks Spotify
        INVALID. Network errors, timeouts, and non-auth HTTP failures are
        inconclusive: they say nothing about whether the credentials are valid,
        so the source is left ENABLED and a genuine problem simply surfaces on
        the first Spotify link. The 10s cap keeps a hung probe from lingering;
        a timeout is treated as inconclusive, not invalid."""
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
        """Return the Spotify client, or raise SpotifyDisabledError if the feature
        is off or its credentials failed startup validation. Call this at every
        Spotify source dispatch: it both narrows the Optional away for the type
        checker and produces the user-facing error, whose text depends on why
        Spotify is unavailable (never configured vs. configured-but-rejected)."""
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
        # Cancel any pending alone-disconnect timer before the atomic gate so it
        # cannot fire after cleanup completes and attempt a second cleanup.
        existing = self._alone_timers.pop(guild.id, None)
        if existing and not existing.done() and existing is not asyncio.current_task():
            existing.cancel()

        # Atomic pop: only the first caller proceeds; any concurrent call (e.g., from
        # on_voice_state_update firing while stop's disconnect is in-flight) gets None
        # and returns immediately, preventing the KeyError TOCTOU race.
        mp = self.mps.pop(guild.id, None)
        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        if mp is None:
            return
        log.info("going to cleanup/disconnect")
        try:
            # Invalidate any in-flight collection drain FIRST — the drain runs
            # in a command task this method does not cancel; every page it
            # commits after this line would land on a guild being torn down.
            # This is the single teardown choke point (-stop, kick,
            # alone-timer, gate timeout, play's error path all funnel here),
            # so one bump covers every door. See GuildQueue.bump_generation.
            await mp.queue.bump_generation()
            # Cancel tasks before disconnecting so the playback loop cannot
            # wake up and start the next song between voice_client.stop() and
            # the task cancellation. VoiceClient.disconnect() calls stop()
            # internally, so audio is silenced when we disconnect below.
            await asyncio.gather(
                cancel_task(mp._prefetch_task),
                cancel_task(mp._progress_task),
                cancel_task(mp._pause_debounce_task),
                cancel_task(mp._player),
                cancel_task(mp._restore_task),
            )
            # Tasks are down — no tick can race this. Dispose of the NP host so
            # no message keeps a mid-song bar frozen by the stop (dedicated NP
            # message → deleted; command-response host → stripped back to its
            # own embeds).
            await mp.retire_np_host_on_stop()
            if guild.voice_client:
                await guild.voice_client.disconnect(force=False)
            # Belt-and-braces: the loop's own CancelledError handler already
            # resets the presence, but only if it was parked inside the block
            # that handles it. Repeat it here — after the disconnect, so this
            # guild's client can no longer register as playing — so a stopped
            # bot never advertises the song it stopped.
            await mp.update_activity(None)
            if mp.store is not None:
                # Intentional stop — clear channel IDs and now-playing state so
                # on_ready does not attempt to recover this guild after restart.
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
        self._active_spans[id(ctx)] = (span, token)

        try:
            if ctx.guild is None:
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
        pair = self._active_spans.pop(id(ctx), None)
        if pair:
            span, token = pair
            span.end()
            otel_context.detach(token)

    async def cog_command_error(self, ctx: commands.Context, error: Exception) -> None:
        """discord.py hook run when a command raises: records the error on the
        active span and, for errors with no other user-visible output, notifies the user.
        """
        # Peek (don't pop) — cog_after_invoke runs in the finally block after this
        # and is responsible for ending the span.
        pair = self._active_spans.get(id(ctx))
        if pair:
            span, _ = pair
            span.record_exception(error)
            span.set_status(StatusCode.ERROR, str(error))

        # validate_commands already sends its own message before raising CommandError,
        # so only handle errors that produce no user-visible output.
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
            # Raised in prepare(), before the command body — so the command's own
            # try/except never sees it (e.g. a second -ping while one is live).
            # Worded off the command name since any future guarded command lands here.
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
        msg = _check_voice_permissions(ctx.author, voice_client, command_name)
        if msg:
            await ctx.send(embed=notice_embed(msg, discord.Color.red()))
            raise commands.CommandError(msg)

    async def _command_error(
        self,
        ctx: commands.Context,
        e: Exception,
        title: str = "Command failed",
    ) -> None:
        # The command's own failure log, folded in here so the 15 command
        # bodies don't each repeat the identical `log.error(f"<cmd> failed:
        # ...")` line before calling this. exc_info=True still captures the
        # live traceback: this runs inside the command's `except` block, so
        # sys.exc_info() is `e`. The command name comes from ctx (the canonical
        # name, matching the literals it replaces) rather than being hand-typed.
        cmd = ctx.command.name if ctx.command else "command"
        log.error(f"{cmd} failed: {type(e).__name__}: {e}", exc_info=True)
        span = trace.get_current_span()
        record_span_error(span, e)  # full detail always goes to the span/logs
        if isinstance(e, ExtractionError):
            # A classified yt-dlp failure: show the user-safe line, not the raw message
            # (which can carry yt-dlp's bug-report boilerplate). See ExtractionError.user_message.
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
    ) -> Union[QueueObject, ResolvedSpotifyStream, ResolvedYoutubePlaylist]:
        """Resolve a parsed URL/search source into something enqueueable.

        Returns a ResolvedSpotifyStream for a Spotify album or playlist (its
        page-1 fetch is already in flight — see the class docstring; no
        ordering decision is made here, so starting the HTTP call early is
        safe), a ResolvedYoutubePlaylist for a YouTube playlist (already
        resolved in full), or a bare QueueObject otherwise.
        """
        if isinstance(source, SpotifySource) and source.type in (
            SpotifyType.ALBUM,
            SpotifyType.PLAYLIST,
        ):
            spotify = self._require_spotify()
            pages = (
                spotify.album_stream(source.id)
                if source.type is SpotifyType.ALBUM
                else spotify.playlist_stream(source.id)
            )
            return ResolvedSpotifyStream(source.type, pages)
        elif isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            if source.list_id is None:
                raise ValueError("YTSource with type=PLAYLIST must have list_id set")
            return ResolvedYoutubePlaylist(
                await YTDL.yt_playlist(source.playlist_url, ctx.author),
                playlist_url=source.playlist_url,
            )
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
            return await YTDL.yt_source(ctx.author, search, ts=ts, redis=self.redis)

    @_tracer.start_as_current_span("bot.enqueue_playlist")
    async def _enqueue_playlist(
        self,
        ctx: commands.Context,
        qobj: ResolvedYoutubePlaylist,
        mp: MusicPlayer,
        *,
        front: bool = False,
    ) -> None:
        """Queue a resolved YouTube playlist in one batch and notify the
        channel. Spotify collections no longer pass through here — they
        stream page-by-page via _begin_stream_enqueue instead."""
        # A playlist front-inserts in FULL, in order — unlike -playnow, which
        # collapses a playlist to its first track. That restriction exists to
        # bound how long an interrupted song waits to return; on this path
        # nothing is playing to interrupt.
        enqueue = mp.queue_put_front if front else mp.queue_put
        tracks = qobj.tracks
        count = len(tracks)
        log.info(f"yt playlist track count: {count}")
        await asyncio.gather(
            send_embed(
                ctx,
                f"Queued playlist — {count} {pluralize(count, 'song')}",
                f"Requested by: [{ctx.author.mention}]\n{qobj.playlist_url}\n\n{queue_message([q.title for q in islice(tracks, 10)])}",
                discord.Color.blue(),
            ),
            enqueue(tracks, prefetch=False),
            ctx.message.add_reaction("👍"),
        )

    async def _acquire_enqueue_slot(
        self, ctx: commands.Context
    ) -> Optional[_GuildEnqueueLock]:
        """Serialize collection enqueues (and -shuffle) per guild.

        Returns the held slot on success — the caller MUST release
        `slot.lock` in a finally — or None when declined, in which case the
        user has already been told why. Best-effort FIFO: asyncio.Lock is
        fair once waiting, and arrival order matches gateway order because
        nothing awaits before this point in the command body. Acquired in the
        body (not a decorator) so ctx.invoke delegation — -playnow's idle
        path invokes play directly, bypassing prepare() — picks it up too.
        """
        assert ctx.guild is not None
        entry = self._enqueue_locks.setdefault(ctx.guild.id, _GuildEnqueueLock())
        if not entry.lock.locked():
            # No await between the locked() check and acquire(): nothing can
            # interleave, so this cannot barge past a live waiter.
            await entry.lock.acquire()
            return entry
        if entry.waiters >= _ENQUEUE_MAX_WAITERS:
            await ctx.send(
                embed=notice_embed(
                    "Too many albums/playlists are queueing right now — try "
                    "again in a moment.",
                    discord.Color.orange(),
                )
            )
            return None
        entry.waiters += 1
        try:
            # A command sitting silently behind a 40s drain reads as a dead
            # bot — acknowledge the wait before blocking.
            await ctx.send(
                embed=notice_embed(
                    "Waiting for another album/playlist to finish queueing…",
                    discord.Color.blue(),
                )
            )
            try:
                async with asyncio.timeout(_ENQUEUE_WAIT_SECS):
                    await entry.lock.acquire()
            except TimeoutError:
                await ctx.send(
                    embed=notice_embed(
                        "Still queueing a large collection — try again in a moment.",
                        discord.Color.orange(),
                    )
                )
                return None
        finally:
            entry.waiters -= 1
        return entry

    @_tracer.start_as_current_span("bot.enqueue_stream")
    async def _begin_stream_enqueue(
        self,
        ctx: commands.Context,
        resolved: ResolvedSpotifyStream,
        mp: MusicPlayer,
        *,
        front: bool,
    ) -> Optional[_StreamDrain]:
        """Enqueue a Spotify collection's page 1 (or, on the buffered path,
        all of it) and send the enqueue embed. Runs INSIDE play's
        AsyncExitStack — on the front path the playback gate is still held, so
        everything here happens before the first note. Returns the drain state
        for _drain_stream_tail, or None when the tail must not run (buffered
        path done, empty collection raised, or the enqueue was refused by a
        concurrent clear/teardown).
        """
        is_album = resolved.kind is SpotifyType.ALBUM
        noun = "album" if is_album else "playlist"
        try:
            page1 = await resolved.first_page
        except StopAsyncIteration:
            # The generators always yield at least one page; this is belt and
            # braces so a future edit can't turn it into an opaque crash.
            raise ValueError(f"The {noun} has no tracks") from None
        if page1.is_last and not page1.titles:
            # Empty collection: raise BEFORE anything is queued — play's error
            # path reports it. A non-last empty page (a playlist whose first
            # 100 items are all episodes) streams on; the drain may still find
            # tracks later.
            raise ValueError(f"The {noun} has no queueable tracks")
        collection = page1.collection
        gen = mp.queue.generation

        if front and await mp.queue.has_restored_backlog():
            # Buffered path (design §3.4 row 3): restored entries exist, so the
            # collection must land AHEAD of them via ONE put_front — successive
            # streamed put_fronts would invert page order. The whole drain runs
            # under the gate hold; this is the one case that keeps the old
            # resolve-everything latency, by design. has_restored_backlog reads
            # the Redis mirror too: qsize()==0 with a mirror ghost must buffer,
            # or the append lands behind an entry the next LPOP wrongly retires.
            titles = list(page1.titles)
            async with contextlib.aclosing(resolved.pages) as pages:
                async for page in pages:
                    titles.extend(page.titles)
            if not titles:
                raise ValueError(f"The {noun} has no queueable tracks")
            ok = await mp.queue_put_front(
                spotify_titles_to_ytsearch(titles),
                prefetch=False,
                expected_generation=gen,
            )
            if not ok:
                log.info("buffered collection enqueue abandoned: queue invalidated")
                return None
            await asyncio.gather(
                self._send_stream_embed(
                    ctx, resolved.kind, collection, titles, enqueued=len(titles)
                ),
                ctx.message.add_reaction("👍"),
            )
            return None

        # Streaming path: page 1 in now, tail after the gate opens. prefetch
        # =False on every page — the default would RPUSH one entry per item
        # (a 100-item page = 100 round-trips) and spawn N prefetch tasks.
        ok = await mp.queue_put(
            spotify_titles_to_ytsearch(page1.titles),
            prefetch=False,
            expected_generation=gen,
        )
        if not ok:
            log.info("stream enqueue abandoned before page 1: queue invalidated")
            return None
        exact = len(page1.titles) if page1.is_last else None
        await asyncio.gather(
            self._send_stream_embed(
                ctx, resolved.kind, collection, page1.titles, enqueued=exact
            ),
            ctx.message.add_reaction("👍"),
        )
        return _StreamDrain(
            resolved=resolved,
            generation=gen,
            enqueued=len(page1.titles),
            total=collection.total,
            # Albums report collection.total upfront (exact — items are never
            # skipped); a multi-page playlist's real count is only known once
            # drained, so it gets a completion notice (L6/G5).
            completion_notice=not is_album and not page1.is_last,
        )

    async def _send_stream_embed(
        self,
        ctx: commands.Context,
        kind: SpotifyType,
        collection: SpotifyCollection,
        titles: list[str],
        *,
        enqueued: Optional[int],
    ) -> None:
        """The enqueue embed for a streamed collection. Albums get their own
        embed — name, artists, count and cover art all ride the same
        GET /v1/albums/{id} call that produced page 1. Playlists cannot have
        name/art (§3.3/M2) and say "~total" until the drain reports the real
        enqueued count — total is an upper bound because episode/null items
        are skipped, never queued."""
        if kind is SpotifyType.ALBUM:
            count = collection.total
            await send_embed(
                ctx,
                truncate_embed_title(
                    f"Queued album — {collection.name or 'Unknown album'}"
                ),
                (
                    f"by {collection.artist_line} · {count} "
                    f"{pluralize(count, 'song')}\n"
                    f"Requested by: [{ctx.author.mention}]\n\n"
                    f"{queue_message(titles)}"
                ),
                discord.Color.blue(),
                thumbnail=collection.thumbnail,
            )
            return
        if enqueued is not None:
            head = f"Queued playlist — {enqueued} {pluralize(enqueued, 'song')}"
            body = f"Requested by: [{ctx.author.mention}]\n\n{queue_message(titles)}"
        else:
            head = "Queued playlist"
            body = (
                f"Requested by: [{ctx.author.mention}]\n"
                f"Queueing ~{collection.total} songs — the rest are on their "
                f"way.\n\n{queue_message(titles)}"
            )
        await send_embed(ctx, head, body, discord.Color.blue())

    @_tracer.start_as_current_span("bot.drain_stream_tail")
    async def _drain_stream_tail(
        self, ctx: commands.Context, mp: MusicPlayer, drain: _StreamDrain
    ) -> None:
        """Drain a stream's remaining pages into the queue, INSIDE the command
        body, after play's AsyncExitStack has exited — the gate is open and
        the first song is already starting, so nothing here is on the
        time-to-first-song path. Runs as the command task on purpose: no
        background task means no sixth cleanup() entry, no cross-coroutine
        lock handoff, and command-error reporting for free (design §3.5).
        background_typing is still active throughout — the typing indicator
        doubling as a "still queueing" signal is intended.

        Every page is a compare-and-put against the generation snapshotted
        before page 1: a -clear or any teardown (stop/kick/alone-timer/gate
        timeout — all bump via cleanup()) refuses the next page atomically,
        and the drain stops quietly. Completion/abandon/failure are reported
        as fresh notice embeds, never by editing the enqueue message — the
        Now Playing host machinery re-edits its host message from its own
        snapshot, so an out-of-band edit of a message it owns would be
        overwritten (and would fight the strip-back on host migration).
        """
        span = trace.get_current_span()
        span.set_attribute("spotify.kind", drain.resolved.kind.value)
        span.set_attribute("spotify.total", drain.total)
        try:
            async with contextlib.aclosing(drain.resolved.pages) as pages:
                async for page in pages:
                    ok = await mp.queue_put(
                        spotify_titles_to_ytsearch(page.titles),
                        prefetch=False,
                        expected_generation=drain.generation,
                    )
                    if not ok:
                        # A clear/teardown landed: the collection this page
                        # belongs to was deleted. Abandon without refilling —
                        # the H2 regression this design exists to prevent.
                        log.info("collection drain abandoned: queue invalidated")
                        with contextlib.suppress(Exception):
                            await ctx.send(
                                embed=notice_embed(
                                    f"Queueing stopped — {drain.enqueued} "
                                    f"{pluralize(drain.enqueued, 'song')} were "
                                    "added before the queue was cleared.",
                                    discord.Color.orange(),
                                )
                            )
                        return
                    drain.enqueued += len(page.titles)
        except Exception:
            # H3 → honest notice, no resume machinery: resume-from-offset is
            # unsound for playlists (an edit shifts every offset — the same
            # §2.5 argument that forbids concurrent playlist fanout). The
            # suppress keeps a failed notice send from masking the real error,
            # which play's error path is about to report.
            with contextlib.suppress(Exception):
                await ctx.send(
                    embed=notice_embed(
                        f"Queued {drain.enqueued} of ~{drain.total} "
                        f"{pluralize(drain.total, 'song')} — the rest failed "
                        f"to load. Re-running the command will re-add the "
                        f"first {drain.enqueued}.",
                        discord.Color.red(),
                    )
                )
            raise
        span.set_attribute("spotify.enqueued", drain.enqueued)
        if drain.completion_notice:
            await ctx.send(
                embed=notice_embed(
                    f"Playlist finished queueing — {drain.enqueued} "
                    f"{pluralize(drain.enqueued, 'song')} in total.",
                    discord.Color.blue(),
                )
            )

    @_tracer.start_as_current_span("bot.enqueue_single")
    async def _enqueue_single(
        self,
        ctx: commands.Context,
        qobj: QueueObject,
        mp: MusicPlayer,
        *,
        front: bool = False,
    ) -> None:
        vc = ctx.voice_client
        if front:
            # The song is about to play, so the "Queued song / Est. playing at"
            # embed below would be wrong — the queue is non-empty here whenever
            # a persisted queue was restored, but those entries are BEHIND this
            # song, not ahead of it. The resume notice replaces it: it names the
            # song starting now (nothing else in this response does — the
            # playback gate is shut, so there is no Now Playing block to host)
            # and adds what the restore knows. Built before the insert, while
            # the queue still holds only the restored entries.
            resume_notice = mp.build_resume_notice_embed(qobj)
            coros: list[Coroutine[Any, Any, Any]] = [
                mp.queue_put_front(qobj),
                ctx.message.add_reaction("👍"),
            ]
            if resume_notice is not None:
                coros.append(ctx.send(embed=resume_notice))
            await asyncio.gather(*coros)
            log.info(f"play (front) qsize: {mp.queue.qsize()}")
            return

        should_show_queued = mp.queue.qsize() > 0 or (
            isinstance(vc, discord.VoiceClient) and vc.is_playing()
        )
        coros: list[Coroutine[Any, Any, Any]] = [
            mp.queue_put(qobj),
            ctx.message.add_reaction("👍"),
        ]
        if should_show_queued:
            coros.append(
                send_embed(
                    ctx,
                    "Queued song",
                    (
                        f"Requested by: [{ctx.author.mention}]\n"
                        f"{qobj.title} - ({qobj.webpage_url})\n"
                        f"Est. playing at {mp.estimated_playing_at()}"
                    ),
                    discord.Color.blue(),
                    thumbnail=qobj.thumbnail,
                )
            )
        await asyncio.gather(*coros)
        log.info(f"play qsize: {mp.queue.qsize()}")

    @commands.command(
        name="play",
        aliases=["p", "sing"],
        brief="queue a song and start playing",
        usage="<url|search>",
        help=(
            "Queues a song and starts playback. Accepts a YouTube link, a YouTube "
            "playlist, a Spotify track, album or playlist link, a SoundCloud link, or "
            "plain words to search YouTube with.\n\n"
            "If the bot is not connected yet it joins your voice channel first. "
            "If something is already playing, the song is appended to the queue "
            "and you get an estimated start time. A YouTube link carrying a "
            "`?t=` / `?ts=` timestamp starts the song at that offset."
        ),
        extras={
            "category": "Playback",
            "examples": [
                "-play never gonna give you up",
                "-play https://youtu.be/dQw4w9WgXcQ?t=43",
                # A real album; the old playlist example was an editorial
                # (37i9…) playlist, which Spotify 404s for third-party apps —
                # a documented example that could never work (§2.3).
                "-play https://open.spotify.com/album/6WgSCcRfaXuBVfM2TpV0Kl",
                "-p https://soundcloud.com/artist/track",
            ],
            "note": (
                "Spotify links are matched to YouTube audio one title at a time. "
                "An album or playlist starts playing after its first page loads "
                "and keeps queueing in the background."
            ),
        },
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.play")
    async def play(self, ctx: commands.Context, url: str) -> None:
        async with background_typing(ctx):
            try:
                # Paused → interject instead of appending. Appending would leave
                # the bot silent with the request buried behind a paused song;
                # -play means "play this". The interrupted song returns PLAYING
                # (resume_paused=False), unlike -playnow which restores it
                # paused. Checked before parse_input so the paused path parses
                # exactly once, inside _interject_flow.
                # docs/PLAY_WHILE_PAUSED_PLAN.md §3.
                paused_vc = ctx.voice_client
                if isinstance(paused_vc, discord.VoiceClient) and paused_vc.is_paused():
                    paused_mp = self.get_mp(ctx)
                    if paused_mp.current_song is not None:
                        return await self._interject_flow(
                            ctx,
                            url,
                            paused_mp,
                            paused_vc,
                            resume_paused=False,
                            require_paused=True,
                        )

                source = parse_input(url, ctx.message.content)

                # Tier-1 admission (design §3.5.1, M5→b): only collections
                # take the per-guild lock — a second collection queueing while
                # one streams would interleave its pages into the first. A
                # single track skips straight through and appends at the
                # current tail, even mid-stream (accepted interleaving; it can
                # also land ahead of a collection still waiting on this lock).
                slot: Optional[_GuildEnqueueLock] = None
                if _is_collection(source):
                    slot = await self._acquire_enqueue_slot(ctx)
                    if slot is None:
                        return  # declined — the user has been told why

                try:
                    await self._play_resolved(ctx, url, source)
                finally:
                    if slot is not None:
                        slot.lock.release()

            except Exception as e:
                await self._command_error(ctx, e, title="Failed to queue song")

    async def _play_resolved(
        self,
        ctx: commands.Context,
        url: str,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> None:
        """The body of -play after parse + lock admission: resolve, enqueue,
        and — for a streamed collection — drain the tail inline. Split out of
        play() so the collection-lock release wraps exactly this much and the
        error handler stays in play()."""
        resolved_stream: Optional[ResolvedSpotifyStream] = None
        drain: Optional[_StreamDrain] = None
        try:
            qobj: Union[QueueObject, ResolvedSpotifyStream, ResolvedYoutubePlaylist]
            async with contextlib.AsyncExitStack() as stack:
                # front: the bot was not connected, so this song jumps ahead
                # of any queue restored from Redis (a -stop leaves its queue
                # persisted). -play on a disconnected bot means "play this",
                # not "play whatever was left over"; the persisted entries
                # resume behind it. docs/PLAYBACK_GATE_PLAN.md.
                front = not ctx.voice_client
                if front:
                    # Hold the playback gate across the join below — join
                    # opens it the moment the handshake lands, which would
                    # start the restored head while queue_source is still
                    # extracting. The hold releases on exiting the stack, after
                    # page 1 / the front insertion — NEVER across the stream
                    # tail, which drains outside the stack with the gate open.
                    await stack.enter_async_context(self.get_mp(ctx).defer_playback())
                    # Launch join concurrently with queue_source — both are pure I/O
                    # (Discord WebSocket handshake vs yt-dlp extraction) with no data
                    # dependency between them. For a Spotify collection the page-1
                    # HTTP call is started inside queue_source as a task, so it
                    # also rides this overlap. await join_task after queue_source
                    # guarantees the voice client is ready before queue_put fires.
                    join_task = asyncio.create_task(ctx.invoke(self.join))
                    try:
                        qobj = await self.queue_source(ctx, source)
                        await join_task
                    except BaseException:
                        if not join_task.done():
                            join_task.cancel()
                            with contextlib.suppress(asyncio.CancelledError, Exception):
                                await join_task
                        # Full cleanup (not just disconnect) — cog_before_invoke already
                        # created a MusicPlayer and started its loop() task. Without
                        # cleanup() that task runs as a zombie for up to 300s waiting on
                        # queue.get(), and store.clear_connection() is never called,
                        # which would trigger spurious crash recovery on restart.
                        # cleanup() also bumps the queue generation, so a
                        # half-started stream of ours is invalidated with it.
                        if ctx.guild is not None:
                            with contextlib.suppress(Exception):
                                await self.cleanup(ctx.guild)
                        raise
                else:
                    qobj = await self.queue_source(ctx, source)

                if isinstance(qobj, ResolvedSpotifyStream):
                    resolved_stream = qobj

                mp = self.get_mp(ctx)
                log.info(f"Voice client: {ctx.voice_client}")

                if front:
                    # Ordering is load-bearing: put_front LPUSHes the Redis
                    # mirror, while restore_entries replays entries that are
                    # already on that list in memory only. Inserting before
                    # restore has read its snapshot double-queues this song.
                    await mp.wait_for_restore()

                if isinstance(qobj, QueueObject):
                    qobj.user_input = url
                    await self._enqueue_single(ctx, qobj, mp, front=front)
                elif isinstance(qobj, ResolvedSpotifyStream):
                    drain = await self._begin_stream_enqueue(ctx, qobj, mp, front=front)
                else:
                    await self._enqueue_playlist(ctx, qobj, mp, front=front)

            # Stack exited: the gate is open and the first song is starting.
            # The tail still runs INSIDE this command — no background task, no
            # sixth cleanup() entry, no cross-coroutine lock handoff (design
            # §3.5); play()'s error handler covers tail failures for free.
            if drain is not None:
                await self._drain_stream_tail(ctx, mp, drain)
        finally:
            if resolved_stream is not None and drain is None:
                # queue_source or _begin_stream_enqueue bailed before the tail
                # took ownership: close the generator and cancel the in-flight
                # page-1 task NOW rather than leaving finalization to asyncgen
                # hooks (see ResolvedSpotifyStream.aclose).
                with contextlib.suppress(Exception):
                    await resolved_stream.aclose()

    async def _resolve_playnow_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> QueueObject:
        """Resolve -playnow input to exactly ONE QueueObject. Collections
        (playlists and albums) collapse to their first track — interjecting a
        whole collection would delay the interrupted song's return
        indefinitely; -play queues the full thing.

        The Spotify path consumes page 1 of the stream and abandons the
        generator under aclosing: exactly one HTTP call, and deliberately no
        cache write (the pagers cache only on a full drain, so a page-1-only
        read can never poison the cache with a truncated collection)."""
        if isinstance(source, SpotifySource) and source.type in (
            SpotifyType.PLAYLIST,
            SpotifyType.ALBUM,
        ):
            is_album = source.type is SpotifyType.ALBUM
            noun = "album" if is_album else "playlist"
            spotify = self._require_spotify()
            stream = (
                spotify.album_stream(source.id)
                if is_album
                else spotify.playlist_stream(source.id)
            )
            async with contextlib.aclosing(stream) as pages:
                page1 = await anext(pages, None)
            titles = page1.titles if page1 is not None else []
            if not titles:
                raise ValueError(f"The {noun} has no queueable tracks")
            await ctx.send(
                embed=notice_embed(
                    f"{noun.capitalize()}s can't be interjected — playing the "
                    f"**first track** now. Use `-play` for the full {noun}.",
                    discord.Color.orange(),
                )
            )
            yts = spotify_titles_to_ytsearch(titles[:1])[0]
            return await YTDL.yt_source(
                ctx.author, yts.ytsearch or "", redis=self.redis
            )
        if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            tracks = await YTDL.yt_playlist(source.playlist_url, ctx.author)
            if not tracks:
                raise ValueError("Playlist has no tracks")
            await ctx.send(
                embed=notice_embed(
                    "Playlists can't be interjected — playing the **first "
                    "track** now. Use `-play` for the full playlist.",
                    discord.Color.orange(),
                )
            )
            return tracks[0]
        qobj = await self.queue_source(ctx, source)
        assert isinstance(qobj, QueueObject)
        return qobj

    @commands.command(
        name="playnow",
        aliases=["pn"],
        brief="play a song immediately, resuming the current one after",
        usage="<url|search>",
        help=(
            "Interrupts whatever is playing so your song starts right now. The "
            "interrupted song is not lost — it comes back from the exact position "
            "it left off at, and if it was paused it returns paused.\n\n"
            "Takes the same input as `-play`. If nothing is playing there is "
            "nothing to interrupt, so this behaves exactly like `-play`.\n\n"
            "A playlist or album can't be interjected — only its **first track** "
            "is played, since queueing the whole thing would delay the "
            "interrupted song indefinitely. Use `-play` for the full collection."
        ),
        extras={
            "category": "Playback",
            "examples": [
                "-playnow never gonna give you up",
                "-pn https://youtu.be/dQw4w9WgXcQ",
            ],
            "note": (
                "`-play` adds to the back of the queue; `-playnow` cuts the line and "
                "hands the current song back afterwards. A song that was nearly over "
                "will not return, and interjecting on top of another `-playnow` song "
                "replaces it rather than stacking."
            ),
        },
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.playnow")
    async def playnow(self, ctx: commands.Context, url: str) -> None:
        async with background_typing(ctx):
            try:
                mp = self.get_mp(ctx)
                vc = ctx.voice_client
                # Nothing live to interrupt → equivalent to -play (also covers
                # not-connected: play joins first). Playlists enqueue in full
                # on this path — interjection semantics don't apply to an
                # idle player (docs/PLAYNOW_PROPOSAL.md §3).
                if (
                    mp.current_song is None
                    or not isinstance(vc, discord.VoiceClient)
                    or not (vc.is_playing() or vc.is_paused())
                ):
                    return await ctx.invoke(self.play, url=url)

                await self._interject_flow(ctx, url, mp, vc)
            except Exception as e:
                await self._command_error(ctx, e, title="Failed to play song now")

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

        Shared by `-playnow` and by `-play` when a song is paused. The two
        differ only in resume_paused: `-playnow` restores exactly what it
        interrupted (paused in → paused out), while `-play` on a paused song
        brings it back playing. Everything else — playlist collapsing, the
        stream warm-up, the ended-mid-resolve fallback, the outcome wording —
        is identical, and duplicating it into `play` would guarantee drift
        (docs/PLAY_WHILE_PAUSED_PLAN.md §4.2).

        require_paused re-reads the pause state after resolution and before
        committing. `-play` only interjects *because* the song is paused, so a
        `-resume` landing during the 1–4s extraction removes the reason — the
        resolved track is appended instead. Reading here rather than at the
        command's entry also means a song that fails to resolve never stops
        the paused song (§3.3).
        """
        source = parse_input(url, ctx.message.content)
        qobj = await self._resolve_playnow_source(ctx, source)
        qobj.user_input = url
        qobj.interjected = True

        # Warm the stream-URL cache BEFORE interrupting the current
        # song — a cache miss at dequeue would otherwise put seconds
        # of yt-dlp dead air between the interrupt and the interjected
        # song starting. Awaited (not spawned like queue_put's
        # warm-up): the current song keeps playing through the wait,
        # which beats stopping it into silence. No-op without Redis;
        # also back-fills duration/thumbnail for the embeds below.
        await YTDL.prefetch_stream(qobj, redis=self.redis)

        if require_paused and not vc.is_paused():
            # Resumed while we were resolving — the reason to interject is
            # gone, so append rather than interrupting a song the user just
            # chose to keep playing. Clear the marker first: a normally queued
            # song must not trigger replace semantics later.
            qobj.interjected = False
            await self._enqueue_single(ctx, qobj, mp)
            return

        outcome = await mp.interject(qobj, vc, resume_paused=resume_paused)
        if outcome is None:
            # The song ended while the input was resolving — nothing
            # to interrupt anymore. The input is already parsed and
            # resolved, so insert the qobj directly rather than
            # re-invoking -play, which would re-parse and re-resolve —
            # and, for a playlist, enqueue ALL tracks right after the
            # first-track-only notice above. FRONT insert, not append:
            # the user asked for "now", and this window can be seconds
            # long (the loop mid-resolve on the next song) with more
            # songs queued behind it. Reset the marker: a normally
            # queued song must not trigger replace semantics when a
            # later interjection interrupts it.
            qobj.interjected = False
            await mp.queue.put_front([qobj])
            await asyncio.gather(
                send_embed(
                    ctx,
                    f"▶️ Playing next: {qobj.title}",
                    f"Requested by: [{ctx.author.mention}]\n"
                    "The song being interrupted already ended — "
                    "queued to play next instead.",
                    discord.Color.blue(),
                    thumbnail=qobj.thumbnail,
                ),
                ctx.message.add_reaction("⏯️"),
            )
            return

        if outcome.replaced:
            desc = (
                f"Replaced **{outcome.interrupted_title}** (also played "
                f"via `-playnow` — it will not return)."
            )
        elif outcome.resume_position is None:
            desc = (
                f"**{outcome.interrupted_title}** was nearly finished "
                f"and will not resume."
            )
        elif outcome.returns_paused:
            # returns_paused, NOT was_paused: with resume_paused=False the song
            # was paused but comes back playing, and "will return paused"
            # would be a lie.
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
        await asyncio.gather(
            send_embed(
                ctx,
                f"▶️ Playing now: {qobj.title}",
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
            # is_playing() is False while paused, so gating on it alone made
            # -skip a total no-op on a paused song — no stop, and not even the
            # reaction (it lived inside the same branch).
            if not (vc.is_playing() or vc.is_paused()):
                return

            # Capture BEFORE stop(): the loop's song-end bookkeeping clears
            # current_song, and the notice below has to name the song that was
            # actually skipped. Primitives, not the object — discord.py's
            # player thread calls source.cleanup() on the way out.
            skipped_title: Optional[str] = None
            skipped_position = ""
            if vc.is_paused():
                song = self.get_mp(ctx).current_song
                if song is not None:
                    skipped_title = song.title
                    # position_secs is frozen while paused, so this is the
                    # exact point the song was left at.
                    skipped_position = fmt_duration(int(song.position_secs))

            vc.stop()

            coros: list[Coroutine[Any, Any, Any]] = []
            if not ctx.invoked_parents:
                coros.append(ctx.message.add_reaction("⏭"))
            if skipped_title is not None:
                # A paused song makes no sound, so stopping it produces no
                # audible cue that the command did anything — unlike an
                # ordinary skip, where the music changing is the feedback.
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
        brief="stop playback, drop the queue and disconnect",
        help=(
            "Stops the current song, discards the queue, removes the Now Playing "
            "card and disconnects the bot from the voice channel.\n\n"
            "This is the full teardown — use `-pause` if you only want to take a "
            "break, or `-clear` if you want to empty the queue but keep playing."
        ),
        extras={"category": "Playback", "examples": ["-stop", "-st"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.stop")
    async def stop(self, ctx: commands.Context) -> None:
        try:
            # Do not call skip before cleanup: skip fires voice_client.stop() which
            # triggers the after callback (play_next.set), giving the playback loop a
            # window to start the next song before the loop task is cancelled.
            # cleanup() cancels _player first, then disconnects — disconnect()
            # internally stops the audio subprocess, so no explicit skip is needed.
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
            "of the channel."
        ),
        extras={"category": "Playback", "examples": ["-resume", "-r"]},
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.resume")
    async def resume(self, ctx: commands.Context) -> None:
        try:
            vc = ctx.voice_client
            if (
                isinstance(vc, discord.VoiceClient)
                and not vc.is_playing()
                and vc.is_paused()
            ):
                mp = self.get_mp(ctx)
                await mp.resume(vc)
                await ctx.message.add_reaction("⏭️")
                # If the -pause confirmation hosts the block, re-host it so
                # "⏸️ Paused at…" becomes plain history instead of sitting
                # beneath a live, advancing bar for the rest of the song.
                await mp.rehost_np_after_resume()
        except Exception as e:
            await self._command_error(ctx, e)

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
            async with background_typing(ctx):
                # Serialized tier (design §3.5.1): shuffle waits for an
                # in-flight collection drain because its semantics need the
                # COMPLETE collection — run mid-stream it would reorder only
                # the pages that have arrived, and the tail would then append
                # in order behind the shuffle, silently half-shuffling. The
                # typical wait is the seconds a few pages take.
                slot = await self._acquire_enqueue_slot(ctx)
                if slot is None:
                    return
                try:
                    await ctx.send(
                        embed=notice_embed(
                            "Please wait... shuffling", discord.Color.blue()
                        )
                    )
                    msg = await mp.queue_shuffle()
                    await ctx.message.add_reaction("🔀")
                    await ctx.send(embed=notice_embed(msg, discord.Color.blue()))
                finally:
                    slot.lock.release()
        except Exception as e:
            await self._command_error(ctx, e)

    @commands.command(
        name="join",
        aliases=["summon"],
        brief="connect the bot to your voice channel",
        help=(
            "Connects the bot to the voice channel you are in and reports its "
            "latency. You rarely need this — `-play` joins for you.\n\n"
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

            # Voice is up — release the playback loop so a persisted queue
            # resumes. No-op while -play holds the gate: it inserts its song at
            # the front first, then opens (docs/PLAYBACK_GATE_PLAN.md §3.3).
            mp.open_playback_gate()

            await asyncio.gather(
                ctx.message.add_reaction("👋"),
                # NOT ctx.invoke(self.ping): that would run the full ~3s health
                # dashboard on every join/cold-play AND skip prepare() (so the
                # ping max_concurrency guard). Cheap one-liner only — §2.3.
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
            cleared = await mp.queue_clear()
            if not cleared:
                await ctx.send(
                    embed=notice_embed(
                        "The queue is already empty.", discord.Color.orange()
                    )
                )
                return
            description = queue_message(cleared)
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
        brief="remove queued songs matching a URL",
        usage="<url>",
        help=(
            "Removes every queued song matching a YouTube URL and reports the "
            "queue positions that were dropped, followed by the updated queue.\n\n"
            "The URL must match the YouTube link shown in the **Now Playing** "
            "card — not the Spotify or search text you originally queued with. "
            "Run it with no URL for a reminder of the format."
        ),
        extras={
            "category": "Queue",
            "examples": ["-remove https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        },
    )
    @commands.before_invoke(validate_commands)
    async def remove(self, ctx: commands.Context, url: Optional[str] = None) -> None:
        if url is None:
            await ctx.send(
                embed=notice_embed(
                    "`-remove <url>` — removes all songs matching the given URL from the queue. "
                    "The URL must match the YouTube link shown in the **Now Playing** embed.",
                    discord.Color.blue(),
                )
            )
            return
        mp = self.get_mp(ctx)
        positions = await mp.queue_remove(url)
        if not positions:
            await send_embed(
                ctx, "", f"No queued songs found matching: <{url}>", discord.Color.red()
            )
            return
        count = len(positions)
        noun = pluralize(count, "song")
        pos_label = pluralize(count, "Position")
        pos_str = ", ".join(str(p) for p in positions)
        await send_embed(
            ctx,
            f"Removed {count} {noun} from the queue",
            "",
            discord.Color.orange(),
            fields=[
                ("URL", f"<{url}>", False),
                (f"{pos_label} removed", pos_str, False),
            ],
        )
        await ctx.send(embed=mp.queue_embed())
        await ctx.message.add_reaction("🗑️")

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
                    # Invoked outside the player's home channel: the host never
                    # leaves home, so answer HERE with a static snapshot (the
                    # MusicContext channel guard keeps it unattached).
                    await ctx.send(embed=mp._build_now_playing_embed(song))
                    return
                # Re-host the live NP block at the bottom of the channel (the
                # old host is retired) instead of sending a static snapshot
                # that immediately goes stale.
                if await mp.repin_now_playing():
                    return
                # Song ended between the liveness check and the repin — fall
                # through to the static/none responses instead of silence.
            if mp.play_message is not None:
                # Crash-recovery window: current_song isn't live yet, but a
                # now-playing snapshot survived the restart. Best-effort static
                # embed (no live progress bar) until loop() starts real playback.
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
        help=(
            "Lists the songs already played in this server, most recent first.\n\n"
            "`--limit N` controls how many are shown (1-50, default 10). History "
            "is stored per server and survives a bot restart."
        ),
        extras={
            "category": "Queue",
            "examples": ["-history", "-h", "-history --limit 25"],
        },
    )
    @commands.before_invoke(validate_commands)
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
            # 8 per message: with the NP block (≤2 embeds) MusicContext.send
            # prepends while a song is live, every chunk stays within Discord's
            # 10-embed cap. Every chunk goes through ctx.send (key invariant #8
            # — never bare channel.send in the player's channel), so the
            # adopt/retire machinery walks the NP block down to the last chunk.
            for start in range(0, len(embeds), HISTORY_EMBEDS_PER_MESSAGE):
                await ctx.send(
                    embeds=embeds[start : start + HISTORY_EMBEDS_PER_MESSAGE]
                )
        except Exception as e:
            await self._command_error(ctx, e)

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
            # It has advertised itself in the help text since early history while doing
            # nothing but replying "currently in development", so the bot promises a
            # feature it does not have. Implementing it is a drain/rotate over
            # GuildQueue, in the same shape as remove().
            # See docs/ARCHITECTURE_PLAN.md §3.8.
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
            if mp.store is not None:
                await mp.store.set_volume(mp.volume)
            await ctx.send(
                embed=notice_embed(
                    f"Set volume to {volume_pct}% (takes effect on next song)",
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
                # Startup validation outcome, not just "is a client configured":
                # it lets the Spotify row say *why* the source is unusable
                # without spending a doomed API call (see probe_spotify).
                spotify_status=self._spotify_status,
                pg_pool=getattr(self, "pg_pool", None),
            )
        except Exception as e:
            await self._command_error(ctx, e)

    # ── Alone-channel disconnect ──────────────────────────────────────────────

    async def _alone_countdown(self, guild: discord.Guild) -> None:
        """Warn the guild's text channel, wait 10s, then disconnect if the
        bot is still alone in its voice channel. Cancelled if a human rejoins."""
        try:
            mp = self.mps.get(guild.id)

            if mp is not None:
                try:
                    # send_with_np (not a bare channel send): this can fire
                    # mid-song, and a bare send would bury the NP host message.
                    embed = discord.Embed(
                        title="No users remaining in voice channel",
                        description="All users have disconnected. The bot will disconnect in **10 seconds** unless someone rejoins.",
                        color=discord.Color.orange(),
                    )
                    await mp.send_with_np(embed=embed)
                except Exception as e:
                    log.warning(
                        f"Failed to send alone-countdown notice in guild {guild.id}: {e}"
                    )

            await asyncio.sleep(10)

            # Span covers only the post-sleep decision so it doesn't stay open for
            # the full 10 seconds (which confuses OTLP exporters and leaks OTel context).
            with _tracer.start_as_current_span(
                "bot.alone_countdown",
                attributes={"discord.guild_id": str(guild.id)},
            ):
                vc = guild.voice_client
                if (
                    isinstance(vc, discord.VoiceClient)
                    and vc.channel is not None
                    and not any(not m.bot for m in vc.channel.members)
                ):
                    log.info(
                        f"Bot still alone in guild {guild.id} after 10s — disconnecting"
                    )
                    await self.cleanup(guild)
        except asyncio.CancelledError:
            pass  # user rejoined or explicit stop; timer was cancelled
        except Exception as e:
            log.error(f"_alone_countdown error in guild {guild.id}: {e}", exc_info=True)
        finally:
            self._alone_timers.pop(guild.id, None)

    # ── Restart recovery listeners ────────────────────────────────────────────

    @commands.Cog.listener()
    async def on_ready(self) -> None:
        """Fires on cold start or session loss (NOT on WebSocket resume).
        Spawns a recovery task per guild so we don't block the event loop."""
        if self.redis is None:
            return
        for guild in self.bot.guilds:
            spawn_background(self._restore_guild(guild), self._restore_tasks)

    @_tracer.start_as_current_span("guild.restore")
    async def _restore_guild(self, guild: discord.Guild) -> None:
        """Attempt to rejoin voice and restore queue for one guild after restart."""
        if self.redis is None:
            return
        if guild.id in self.mps:
            return

        store = GuildRedisStore(self.redis, guild.id)

        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        # Distributed lock prevents two bot instances from racing on the same guild.
        # Acquired inside the span so the SET NX EX Redis call is a child span.
        if not await store.acquire_recovery_lock():
            trace.get_current_span().set_attribute("restore.skipped_lock", True)
            log.info(
                f"Recovery lock held by another instance for guild {guild.id}, skipping"
            )
            return
        try:
            # One pipelined read serves every gate below: the connection gate
            # (state hash) and the anything-to-restore gate (queue length +
            # crashed song). Only the queue *length* is read here — the actual
            # queue/now-playing/history payload is re-read by _restore_state
            # after a successful connect, so a stopped guild's leftover queue
            # never rides the wire on the common "nothing to do" path.
            gate = await store.get_recovery_gate()
            if gate is None:
                # Redis read failed — do NOT treat as "nothing to restore". Skip
                # this attempt; the recovery lock expires in 60s and the next
                # on_ready (session re-establishment) retries.
                log.warning(f"Recovery skipped for guild {guild.id}: state read failed")
                return
            guild_state = gate.state
            # Equivalent to `not guild_state.has_active_connection`, spelled as
            # explicit None checks so the channel IDs narrow to int below.
            vc_id = guild_state.voice_channel_id
            tc_id = guild_state.text_channel_id
            if vc_id is None or tc_id is None:
                return

            voice_channel = guild.get_channel(vc_id)
            text_channel = guild.get_channel(tc_id)
            voice_ok = isinstance(voice_channel, discord.VoiceChannel)
            text_ok = isinstance(text_channel, discord.TextChannel)

            if not voice_ok or not text_ok:
                # Clear stale channel IDs so this guild is not re-attempted on every reconnect.
                await store.clear_connection()
                trace.get_current_span().set_attribute("restore.channel_missing", True)
                log.warning(
                    f"Recovery skipped for guild {guild.id}: "
                    f"voice_channel_id={vc_id} (resolved={voice_ok}) "
                    f"text_channel_id={tc_id} (resolved={text_ok})"
                )

                notify_channel: Optional[discord.TextChannel] = None
                if text_ok:
                    notify_channel = text_channel
                elif guild.me is not None:
                    if (
                        guild.system_channel is not None
                        and guild.system_channel.permissions_for(guild.me).send_messages
                    ):
                        notify_channel = guild.system_channel
                    else:
                        notify_channel = next(
                            (
                                ch
                                for ch in guild.text_channels
                                if ch.permissions_for(guild.me).send_messages
                            ),
                            None,
                        )

                if notify_channel is not None:
                    deleted: list[str] = []
                    if not voice_ok:
                        deleted.append("voice channel")
                    if not text_ok:
                        deleted.append("text channel")
                    what = " and ".join(deleted)
                    verb = "was" if len(deleted) == 1 else "were"
                    try:
                        await notify_channel.send(
                            embed=notice_embed(
                                f"⚠️ I came back online but the {what} I was playing in "
                                f"{verb} deleted. Use `-play` in a voice channel to start fresh.",
                                discord.Color.orange(),
                            )
                        )
                    except Exception as notify_err:
                        log.warning(
                            f"Failed to send channel-deleted notification for "
                            f"guild {guild.id}: {notify_err}"
                        )
                return

            # Check there is something to restore before connecting.
            if not gate.has_restorable_playback:
                return

            trace.get_current_span().set_attribute(
                "restore.queue_count", gate.pending_count
            )
            trace.get_current_span().set_attribute(
                "restore.crashed_song", guild_state.has_crashed_song
            )

            try:
                await voice_channel.connect(timeout=30.0, reconnect=True)
                await guild.change_voice_state(
                    channel=voice_channel, self_mute=False, self_deaf=True
                )
            except Exception as e:
                trace.get_current_span().set_attribute(
                    "restore.voice_connect_failed", True
                )
                trace.get_current_span().record_exception(e)
                trace.get_current_span().set_status(
                    StatusCode.ERROR, f"voice connect failed: {e}"
                )
                log.warning(f"Could not rejoin voice for guild {guild.id}: {e}")
                return

            mp = MusicPlayer(self.bot, guild, text_channel, self, redis=self.redis)
            mp.start()
            self.mps[guild.id] = mp

            log.info(
                f"Restored guild {guild.id} in #{text_channel.name} / {voice_channel.name}"
            )
        except Exception as e:
            record_span_error(trace.get_current_span(), e)
            log.error(f"_restore_guild failed for guild {guild.id}: {e}", exc_info=True)
        finally:
            await store.release_recovery_lock()

    @commands.Cog.listener()
    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Handles two cases: the bot itself being disconnected/moved (full
        cleanup or stale-timer cancellation) and a human member's channel
        change relative to the bot's current channel (starts/cancels the
        10s alone-disconnect countdown)."""
        guild = member.guild

        # ── Case A: bot itself was disconnected or moved ──────────────────────
        if self.bot.user is not None and member.id == self.bot.user.id:
            if before.channel is not None and after.channel is None:
                # Bot ejected — full cleanup.
                if guild.id in self.mps:
                    with _tracer.start_as_current_span(
                        "bot.voice_state_update",
                        attributes={"discord.guild_id": str(guild.id)},
                    ):
                        log.info(
                            f"Bot disconnected from voice in guild {guild.id}, cleaning up"
                        )
                        await self.cleanup(guild)
            elif before.channel is not None and after.channel is not None:
                # Bot moved to a different channel — cancel any stale alone-timer
                # that was counting down for the old channel.
                existing = self._alone_timers.pop(guild.id, None)
                if existing and not existing.done():
                    existing.cancel()
            return

        # ── Case B: a human member's voice state changed ──────────────────────
        if guild.id not in self.mps:
            return  # bot isn't active in this guild

        vc = guild.voice_client
        if not isinstance(vc, discord.VoiceClient) or vc.channel is None:
            return

        # Skip mute/deafen/server-deafen events — channel is unchanged.
        if before.channel == after.channel:
            return

        # Only care about events that affect the bot's current channel.
        if before.channel != vc.channel and after.channel != vc.channel:
            return

        human_members = [m for m in vc.channel.members if not m.bot]

        if len(human_members) == 0:
            # Bot is now alone — start (or restart) the 10-second countdown.
            existing = self._alone_timers.pop(guild.id, None)
            if existing and not existing.done():
                existing.cancel()
            log.info(f"Bot is alone in guild {guild.id}, starting 10s disconnect timer")
            self._alone_timers[guild.id] = asyncio.create_task(
                self._alone_countdown(guild)
            )
        else:
            # A human is present — cancel any running alone-timer.
            existing = self._alone_timers.pop(guild.id, None)
            if existing and not existing.done():
                log.info(f"User rejoined guild {guild.id}, cancelling alone timer")
                existing.cancel()


async def setup(bot: commands.Bot) -> None:
    await bot.add_cog(MusicBot(bot))
