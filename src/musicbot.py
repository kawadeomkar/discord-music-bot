import asyncio
import contextlib
import time
from dataclasses import dataclass
from itertools import islice
from typing import (
    Any,
    Optional,
    Union,
    assert_never,
)
from collections.abc import AsyncGenerator, Coroutine, Sequence

import discord
from discord.ext import commands

import redis.asyncio as aioredis

from src.config import (
    ENVIRONMENT,
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
    debug_mode_default,
    spotify_enabled,
)
from src import debug as debug_mode
from src import leaderboard
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
    read_guild_configs,
)
from src.sources import (
    SoundcloudSource,
    SpotifySource,
    SpotifyType,
    YTSource,
    YTType,
    parse_input,
    query_source_of,
    spotify_playlist_to_ytsearch,
)
from src.spotify import Spotify, SpotifyAuthError
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


HISTORY_MIN_LIMIT = 1
# Pinned to HISTORY_CACHE_LIMIT. recent() serves this command from the Redis list
# alone, which holds exactly that many entries, so a larger ceiling here returns a
# short page instead of failing. Raise both together or neither.
HISTORY_MAX_LIMIT = HISTORY_CACHE_LIMIT
# 8 song embeds + the ≤2-embed NP block MusicContext.send may prepend = Discord's
# per-message cap of 10, so the block always fits and is never shed.
HISTORY_EMBEDS_PER_MESSAGE = 8


class HistoryFlags(commands.FlagConverter, prefix="--", delimiter=" "):
    limit: int = 10


def _stamp_query_source(qobjs: Sequence[QueueObject], token: str) -> None:
    """Write the parse-time classification onto resolved songs.

    Applied after resolution rather than threaded through yt_source/yt_playlist:
    those are the extraction API, and this is the same seam MusicPlayer uses to
    copy the enqueue stamps onto what yt_source builds."""
    for qobj in qobjs:
        qobj.query_source = token


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
    """A YouTube playlist already resolved to playable QueueObjects."""

    tracks: list[QueueObject]


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
    # Exception only, not CancelledError: background_typing() cancels this on the way
    # out, and letting that propagate is what marks the task genuinely cancelled
    # rather than completed. Swallowing it would stop a shutdown at this frame.
    except Exception:
        pass  # cosmetic — never let typing failures surface


@contextlib.asynccontextmanager
async def background_typing(ctx: commands.Context) -> AsyncGenerator[None]:
    """Non-blocking ctx.typing(): the first POST /typing runs in a background task so
    the command body starts immediately, and the keepalive is cancelled when the body
    finishes. The whole CM lives inside the task — never enter/exit Typing manually
    across tasks."""
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
        "history_archive",
        "_active_spans",
        "_alone_timers",
        "_restore_tasks",
        "_debug_default",
        "_debug_overrides",
        "_debug_toggle_seq",
        "_debug_toggled_at",
        "_debug_unpersisted",
        "_runtime_sampler",
    )

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
        # id(ctx) → the in-flight command's span, otel token and start time.
        self._active_spans: dict[int, ActiveCommand] = {}
        self._alone_timers: dict[int, asyncio.Task] = {}
        self._restore_tasks: set[asyncio.Task] = set()
        # Debug mode. The env var is read ONCE, here, which is what makes a garbage
        # value abort startup inside load_extension rather than surfacing later from
        # somewhere that swallows it. Overrides are per guild — an ungated toggle's
        # blast radius must stay inside the guild that typed it — and DURABLE: the
        # choice lives in guild:{id}:config and outlives restarts, so this dict is a
        # read cache hydrated from Redis, not the record itself. A guild that never
        # chose is absent from it and follows _debug_default, and keeps following it
        # when the operator changes the env var.
        self._debug_default: bool = debug_mode_default()
        self._debug_overrides: dict[int, bool] = {}
        # Monotonic stamp bumped by every explicit toggle, plus the value it had when
        # each guild was last toggled. _load_debug_overrides reads Redis and applies
        # the result across an await; these let it tell whether a guild changed under
        # it in that window and leave the newer value alone. One int per guild that
        # has ever toggled, pruned alongside the override on guild removal.
        self._debug_toggle_seq: int = 0
        self._debug_toggled_at: dict[int, int] = {}
        # Guilds whose cached value did NOT reach Redis, so -debug can report
        # "this session only" rather than the durability the toggle already warned
        # did not happen. Cleared as soon as a write or a hydration proves the
        # durable copy agrees.
        self._debug_unpersisted: set[int] = set()
        # One sampler per cog, never module state: a module global outlives a cog
        # reload and would leak its task. cog_load is what starts it (see
        # RuntimeSampler.apply for why load, not only toggles).
        self._runtime_sampler = debug_mode.RuntimeSampler()
        if self._debug_default and ENVIRONMENT == "production":
            # Debug modes announce themselves in production; the convention is
            # Flask's and Django's. Observation-only, so this is an advisory, not
            # a refusal — but a production deployment should have chosen it.
            log.warning(
                "DEBUG_MODE is on in production: every command response will "
                "carry a debug footer (trace id, timings, runtime metrics). "
                "Nothing about playback changes. Unset DEBUG_MODE to turn it off."
            )

    async def cog_load(self) -> None:
        """Kick off Spotify credential validation without blocking startup.
        discord.py awaits this inside setup_hook, before the bot connects, so
        anything awaited here delays it. The probe is a live network call, spawned
        fire-and-forget; _spotify_status stays optimistically enabled meanwhile."""
        # At load, not only on toggles — RuntimeSampler.apply's docstring has the
        # reason. _load_debug_overrides re-syncs it once the stored choices land.
        self._sync_runtime_sampler()
        spawn_background(self._load_debug_overrides(), self._restore_tasks)
        if self.spotify is None:
            return
        spawn_background(self._validate_spotify_credentials(), self._restore_tasks)

    async def cog_unload(self) -> None:
        """Stop the runtime sampler unconditionally. A reload that left it running
        would drip /proc reads for the life of the process.

        The background tasks go FIRST: _load_debug_overrides ends in
        _sync_runtime_sampler, so one still in flight would restart the sampler
        straight after aclose() and leak it, holding a dead cog alive.
        """
        await asyncio.gather(*(cancel_task(t) for t in list(self._restore_tasks)))
        await self._runtime_sampler.aclose()

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
        # cannot fire after cleanup completes and attempt a second one.
        existing = self._alone_timers.pop(guild.id, None)
        if existing and not existing.done() and existing is not asyncio.current_task():
            existing.cancel()

        # Atomic pop: only the first caller proceeds. A concurrent call (e.g.
        # on_voice_state_update firing while stop's disconnect is in flight) gets
        # None and returns, avoiding the KeyError TOCTOU race.
        mp = self.mps.pop(guild.id, None)
        trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
        if mp is None:
            return
        log.info("going to cleanup/disconnect")
        try:
            # Cancel tasks before disconnecting so the loop cannot wake and start
            # the next song between voice_client.stop() and cancellation.
            # disconnect() calls stop() internally, silencing audio below.
            await asyncio.gather(
                cancel_task(mp._prefetch_task),
                cancel_task(mp._progress_task),
                cancel_task(mp._pause_debounce_task),
                cancel_task(mp._player),
                cancel_task(mp._restore_task),
            )
            # Tasks are down, so no tick can race this. Dispose of the NP host so
            # no message keeps a bar frozen mid-song by the stop.
            await mp.retire_np_host_on_stop()
            if guild.voice_client:
                await guild.voice_client.disconnect(force=False)
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
        msg = _check_voice_permissions(ctx.author, voice_client, command_name)
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
            if isinstance(e, ExtractionError):
                # Show the user-safe line, not the raw message, which can carry
                # yt-dlp's bug-report boilerplate. See ExtractionError.user_message.
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
    ) -> Union[QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist]:
        """Resolve a parsed URL/search source into something enqueueable: a
        ResolvedSpotifyPlaylist (titles still needing per-title YouTube resolution),
        a ResolvedYoutubePlaylist (already resolved), or a bare QueueObject."""
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            # Titles, not QueueObjects — spotify_playlist_to_ytsearch stamps the
            # YTSources they become.
            return ResolvedSpotifyPlaylist(
                await self._require_spotify().playlist(source.id)
            )
        elif isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            if source.list_id is None:
                raise ValueError("YTSource with type=PLAYLIST must have list_id set")
            tracks = await YTDL.yt_playlist(source.playlist_url, ctx.author)
            _stamp_query_source(tracks, query_source_of(source))
            return ResolvedYoutubePlaylist(tracks)
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
            qobj = await YTDL.yt_source(ctx.author, search, ts=ts, redis=self.redis)
            _stamp_query_source([qobj], query_source_of(source))
            return qobj

    @_tracer.start_as_current_span("bot.enqueue_playlist")
    async def _enqueue_playlist(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
        qobj: Union[ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist],
        mp: MusicPlayer,
        *,
        front: bool = False,
    ) -> None:
        """Queue a resolved playlist and notify the channel — branches on the
        resolved shape since Spotify playlists arrive as titles needing YouTube
        search resolution while YouTube playlists arrive pre-resolved."""
        # A playlist front-inserts in full, in order — unlike -playnow, which
        # collapses it to the first track to bound how long an interrupted song
        # waits. Nothing is playing to interrupt on this path.
        enqueue = mp.queue_put_front if front else mp.queue_put
        if isinstance(qobj, ResolvedSpotifyPlaylist):
            titles = qobj.titles
            qobjs_yt = spotify_playlist_to_ytsearch(titles)
            log.info(f"ytsearch qobjs: {qobjs_yt}")
            await asyncio.gather(
                send_embed(
                    ctx,
                    "Queued playlist",
                    f"Requested by: [{ctx.author.mention}]\n\n{queue_message(titles)}",
                    discord.Color.blue(),
                ),
                enqueue(qobjs_yt, prefetch=False),
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
            await asyncio.gather(
                send_embed(
                    ctx,
                    f"Queued playlist — {count} {pluralize(count, 'song')}",
                    f"Requested by: [{ctx.author.mention}]\n{playlist_url}\n\n{queue_message([q.title for q in islice(tracks, 10)])}",
                    discord.Color.blue(),
                ),
                enqueue(tracks, prefetch=False),
                ctx.message.add_reaction("👍"),
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
            "playlist, a Spotify track or playlist link, a SoundCloud link, or "
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
                "-play https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
                "-p https://soundcloud.com/artist/track",
            ],
            "note": (
                "Spotify links are matched to YouTube audio one title at a time, "
                "so a long playlist takes a few seconds to finish queueing."
            ),
        },
    )
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.play")
    async def play(self, ctx: commands.Context, url: str) -> None:
        async with background_typing(ctx):
            try:
                # Paused → interject, not append: appending leaves the bot silent
                # with the request buried behind a paused song. The interrupted song
                # returns PLAYING, unlike -playnow. Checked before parse_input so the
                # paused path parses once, inside _interject_flow.
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

                qobj: Union[
                    QueueObject, ResolvedSpotifyPlaylist, ResolvedYoutubePlaylist
                ]
                async with contextlib.AsyncExitStack() as stack:
                    # front: not connected, so this song jumps ahead of any queue
                    # restored from Redis (a -stop leaves its queue persisted).
                    # -play on a disconnected bot means "play this", not "play
                    # the leftovers".
                    front = not ctx.voice_client
                    if front:
                        # Hold the gate across the join below: join opens it the
                        # moment the handshake lands, which would start the
                        # restored head while queue_source is still extracting.
                        # Released on exiting the stack, after the front insertion.
                        await stack.enter_async_context(
                            self.get_mp(ctx).defer_playback()
                        )
                        # Concurrent with queue_source: both are pure I/O (voice
                        # handshake vs yt-dlp extraction) with no data dependency.
                        # Awaiting join_task after queue_source guarantees the voice
                        # client is ready before queue_put fires.
                        join_task = asyncio.create_task(ctx.invoke(self.join))
                        try:
                            qobj = await self.queue_source(ctx, source)
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
                            if ctx.guild is not None:
                                with contextlib.suppress(Exception):
                                    await self.cleanup(ctx.guild)
                            raise
                    else:
                        qobj = await self.queue_source(ctx, source)

                    mp = self.get_mp(ctx)
                    log.info(f"Voice client: {ctx.voice_client}")

                    if front:
                        # Order matters: put_front LPUSHes the mirror, while
                        # restore_entries replays already-listed entries in memory
                        # only, so inserting before restore reads its snapshot would
                        # double-queue this song.
                        await mp.wait_for_restore()

                    if isinstance(qobj, QueueObject):
                        qobj.user_input = url
                        await self._enqueue_single(ctx, qobj, mp, front=front)
                    else:
                        await self._enqueue_playlist(ctx, source, qobj, mp, front=front)

            except Exception as e:
                await self._command_error(ctx, e, title="Failed to queue song")

    async def _resolve_playnow_source(
        self,
        ctx: commands.Context,
        source: Union[SpotifySource, YTSource, SoundcloudSource],
    ) -> QueueObject:
        """Resolve -playnow input to exactly one QueueObject. Playlists collapse to
        their first track — interjecting a whole one would delay the interrupted
        song's return indefinitely (use -play)."""
        playlist_notice = notice_embed(
            "Playlists can't be interjected — playing the **first track** now. "
            "Use `-play` for the full playlist.",
            discord.Color.orange(),
        )
        if isinstance(source, SpotifySource) and source.type == SpotifyType.PLAYLIST:
            titles = await self._require_spotify().playlist(source.id)
            if not titles:
                raise ValueError("Playlist has no tracks")
            await ctx.send(embed=playlist_notice)
            yts = spotify_playlist_to_ytsearch(titles[:1])[0]
            # Both playlist branches resolve directly rather than through
            # queue_source, so each stamps its own result.
            qobj = await YTDL.yt_source(
                ctx.author, yts.ytsearch or "", redis=self.redis
            )
            _stamp_query_source([qobj], query_source_of(yts))
            return qobj
        if isinstance(source, YTSource) and source.type == YTType.PLAYLIST:
            tracks = await YTDL.yt_playlist(source.playlist_url, ctx.author)
            if not tracks:
                raise ValueError("Playlist has no tracks")
            await ctx.send(embed=playlist_notice)
            _stamp_query_source(tracks[:1], query_source_of(source))
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
            "A playlist can't be interjected — only its **first track** is played, "
            "since queueing the whole thing would delay the interrupted song "
            "indefinitely. Use `-play` for the full playlist."
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
                # Nothing live to interrupt → equivalent to -play (which also
                # covers not-connected, since play joins first). Playlists enqueue
                # in full here: interjection semantics don't apply to an idle
                # player.
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

        Shared by `-playnow` and by `-play` on a paused song; they differ only in
        resume_paused (`-playnow` restores paused-in → paused-out, `-play` brings it
        back playing). require_paused re-reads the pause state after resolution,
        before committing: `-play` interjects only *because* the song is paused, so a
        `-resume` landing during the 1–4s extraction removes the reason and the track
        is appended instead. Reading it here rather than at command entry also means
        a song that fails to resolve never stops the paused song.
        """
        source = parse_input(url, ctx.message.content)
        qobj = await self._resolve_playnow_source(ctx, source)
        qobj.user_input = url
        qobj.interjected = True

        # Warm the stream-URL cache before interrupting: a cache miss at dequeue puts
        # seconds of yt-dlp dead air between the interrupt and the new song. Awaited,
        # not spawned — the current song plays through the wait. No-op without Redis;
        # also back-fills duration/thumbnail for the embeds below.
        await YTDL.prefetch_stream(qobj, redis=self.redis)

        if require_paused and not vc.is_paused():
            # Resumed during the resolve — the reason to interject is gone, so
            # append rather than interrupt a song the user just chose to keep
            # playing. Clear the marker: a normally queued song must not trigger
            # replace semantics later.
            qobj.interjected = False
            await self._enqueue_single(ctx, qobj, mp)
            return

        outcome = await mp.interject(qobj, vc, resume_paused=resume_paused)
        if outcome is None:
            # The song ended during the resolve — nothing left to interrupt. Insert
            # qobj directly rather than re-invoking -play, which would re-parse,
            # re-resolve and (for a playlist) enqueue all tracks right after the
            # first-track-only notice above. Front, not append: the user asked for
            # "now", and this window can be seconds long with songs queued behind.
            # Reset the marker or a normally queued song triggers replace semantics.
            qobj.interjected = False
            # The player's wrapper, not queue.put_front directly: it stamps the
            # enqueue under the queue mutex like every other user-facing insert.
            # prefetch=False — the stream URL was warmed above.
            await mp.queue_put_front(qobj, prefetch=False)
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
            # is_playing() is False while paused, so gating on it alone made -skip
            # a total no-op on a paused song — not even the reaction.
            if not (vc.is_playing() or vc.is_paused()):
                return

            # Capture before stop(): the loop's song-end bookkeeping clears
            # current_song, and the notice must name the song actually skipped.
            # Primitives, not the object — the player thread calls cleanup() on it.
            skipped_title: Optional[str] = None
            skipped_position = ""
            if vc.is_paused():
                song = self.get_mp(ctx).current_song
                if song is not None:
                    skipped_title = song.title
                    # position_secs is frozen while paused: the exact leave point.
                    skipped_position = fmt_duration(int(song.position_secs))

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
                # "⏸️ Paused at…" becomes history instead of sitting beneath a
                # live, advancing bar for the rest of the song.
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
                # The startup validation outcome, not just "is a client
                # configured": lets the Spotify row say *why* the source is
                # unusable without spending a doomed API call (see probe_spotify).
                spotify_status=self._spotify_status,
                archive=self.history_archive,
            )
        except Exception as e:
            await self._command_error(ctx, e)

    # ── Debug mode ────────────────────────────────────────────────────────────

    def debug_enabled(self, guild_id: Optional[int]) -> bool:
        """Whether debug mode is on for this guild — the one query surface for it.

        DEBUG_MODE is the host-wide default. It is read SYNCHRONOUSLY and from
        memory on purpose: MusicContext.send calls this on every reply, so anything
        that needed IO here would put it on the hot path of every command.

        DEBUG_MODE is the default every guild starts from: set it and all of them
        are on unless they opted out; leave it false or unset and each guild turns
        itself on with `-debug --enable`. A guild with no stored choice (and every
        DM, which has no guild to scope one to) follows that default.

        SYNCHRONOUS and in-memory on purpose: MusicContext.send calls this on every
        reply, so a Redis round trip here would put the persistence layer on the hot
        path of every command. Redis is the durable copy; `_debug_overrides` is the
        read cache, hydrated at cog_load and on_ready and written through by the
        toggle.
        """
        if guild_id is None:
            return self._debug_default
        return self._debug_overrides.get(guild_id, self._debug_default)

    async def _load_debug_overrides(self) -> None:
        """Hydrate the in-memory cache from each guild's stored config.

        Runs at cog_load and again on every on_ready — reconnects included, and
        on_ready re-fires on every session loss, not once per process. Two skip rules
        make replaying it safe, and both are load-bearing:

        A guild whose config could not be READ is skipped entirely. read_guild_configs
        omits it rather than handing back a zero value, because "Redis blinked" and
        "this guild never chose" must not be the same answer here — treating them
        alike made one failed read DELETE a correct stored choice and revert that
        guild to the host default for the rest of the process. This is the discipline
        _restore_guild already applies to a failed get_recovery_gate.

        A guild toggled while this pass was reading is skipped too. The read and the
        apply straddle an await, so a `-debug --enable` landing between them would
        otherwise be overwritten by the value that was true before the user ran it:
        they are told "saved for this server", Redis agrees, and the footer never
        appears until the next session loss.
        """
        if self.redis is None:
            return
        guilds = list(self.bot.guilds)
        if not guilds:
            return
        started = self._debug_toggle_seq
        configs = await read_guild_configs(self.redis, [g.id for g in guilds])
        for guild in guilds:
            config = configs.get(guild.id)
            if config is None or self._debug_toggled_at.get(guild.id, 0) > started:
                continue
            if config.debug_mode is None:
                # No stored choice: follow the host default, and do NOT cache that
                # — caching it would freeze the guild against a later env change.
                self._debug_overrides.pop(guild.id, None)
            else:
                self._debug_overrides[guild.id] = config.debug_mode
                # Read back from Redis, so the durable copy is the source.
                self._debug_unpersisted.discard(guild.id)
        self._sync_runtime_sampler()

    @property
    def runtime_snapshot(self) -> Optional["debug_mode.RuntimeSnapshot"]:
        """The rolling runtime metrics the debug footer prints, or None before the
        sampler's first tick. Read by MusicContext.send."""
        return self._runtime_sampler.snapshot

    def _sync_runtime_sampler(self) -> None:
        """Run the sampler exactly while some guild is effectively debug-enabled."""
        self._runtime_sampler.apply(
            wanted=self._debug_default or any(self._debug_overrides.values())
        )

    @commands.Cog.listener()
    async def on_guild_remove(self, guild: discord.Guild) -> None:
        """Forget a departed guild's debug override.

        One bool per guild, so this is hygiene rather than a leak — but the override
        is the only debug state that is NOT re-derived from the environment, so a
        guild that removes the bot and re-adds it inside one process lifetime would
        silently resume its old setting, which reads as the toggle ignoring them.
        Re-syncing the sampler matters too: the last enabled guild leaving should
        stop it, exactly as `--disable` would.
        """
        self._debug_toggled_at.pop(guild.id, None)
        self._debug_unpersisted.discard(guild.id)
        if self._debug_overrides.pop(guild.id, None) is not None:
            self._sync_runtime_sampler()
        # The durable copy goes too, or a guild that removes and re-adds the bot
        # silently resumes a setting nobody there chose — and the key would sit in
        # Redis forever, since config carries no TTL by design.
        if self.redis is not None:
            await GuildRedisStore(self.redis, guild.id).clear_config()

    @commands.command(
        name="debug",
        aliases=["dbg"],
        brief="show or toggle debug mode",
        usage="[--enable | --disable]",
        help=(
            "Reports whether debug mode is on for this server, and where that came "
            "from — a choice saved here, or the host's default.\n\n"
            "`--enable` turns it on for this server, which adds a footer carrying "
            "the trace id and timing to every reply — paste that id to the operator "
            "and they can find the exact request in the logs. `--disable` turns it "
            "back off. The choice is saved for this server and survives restarts; a "
            "server that has never set it follows the host's default. Toggling needs "
            "the **Manage Server** permission."
        ),
        extras={
            "category": "Utility",
            # Read by cog_before_invoke to skip get_mp(): this command reports on a
            # guild, so manufacturing a player to look at would be the observer
            # changing what it observes.
            "observation_only": True,
            "examples": ["-debug", "-debug --enable", "-debug --disable"],
            "note": (
                "Debug mode is per server and only changes what is DISPLAYED — "
                "never how the bot plays, queues or stores anything."
            ),
        },
    )
    # No validate_commands: diagnosing a bot needs no voice channel (-ping's
    # precedent). One in flight per guild, wait=False; cog_command_error renders
    # the refusal.
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
            guild_id = ctx.guild.id if ctx.guild else None
            on = self.debug_enabled(guild_id)
            source = debug_mode.mode_source(
                guild_id is not None and guild_id in self._debug_overrides,
                persisted=guild_id not in self._debug_unpersisted,
            )
            await ctx.send(
                embed=notice_embed(
                    f"Debug mode is **{'on' if on else 'off'}** here ({source}). "
                    "Replies "
                    + ("carry" if on else "do not carry")
                    + " a footer with the trace id and timing.",
                    discord.Color.blue(),
                )
            )
        except Exception as e:
            await self._command_error(ctx, e)

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
                    f"**{'on' if self._debug_default else 'off'}** here, following "
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
                    "Debug mode changes what **every** reply in this server looks "
                    "like, so switching it needs the Manage Server permission. Run "
                    "`-debug` on its own to see the current state.",
                    discord.Color.red(),
                )
            )
            return
        enabled = action is debug_mode.DebugAction.ENABLE
        # Redis FIRST, cache second. The reverse would let a failed write leave the
        # cache claiming a setting the durable copy never took, which the next
        # on_ready would silently undo.
        persisted = False
        if self.redis is not None:
            persisted = await GuildRedisStore(self.redis, ctx.guild.id).set_debug_mode(
                enabled
            )
        self._debug_overrides[ctx.guild.id] = enabled
        # Stamp this guild so a hydration pass that read BEFORE this write cannot
        # apply its older value on top of it — see _load_debug_overrides.
        self._debug_toggle_seq += 1
        self._debug_toggled_at[ctx.guild.id] = self._debug_toggle_seq
        if persisted:
            self._debug_unpersisted.discard(ctx.guild.id)
        else:
            self._debug_unpersisted.add(ctx.guild.id)
        # Re-evaluated on every toggle: the sampler runs only while some guild
        # wants it, so the last --disable stops it.
        self._sync_runtime_sampler()
        log.info(
            f"debug mode {'enabled' if enabled else 'disabled'} by command",
            persisted=persisted,
        )
        # Say which kind of change this was. A guild told "on" that quietly reverts
        # on the next restart reads as the bot ignoring them, so a degraded write is
        # named rather than rounded up to success.
        durability = (
            "The setting is saved for this server."
            if persisted
            else "⚠️ It could not be saved (Redis is unavailable), so it applies "
            "until the bot restarts."
        )
        await ctx.send(
            embed=notice_embed(
                f"Debug mode is now **{'on' if enabled else 'off'}** for this "
                "server. Replies "
                + ("carry" if enabled else "no longer carry")
                + " a debug footer; nothing about playback changes either way. "
                + durability,
                discord.Color.blue(),
            )
        )

    # ── Alone-channel disconnect ──────────────────────────────────────────────

    async def _alone_countdown(self, guild: discord.Guild) -> None:
        """Warn the guild's text channel, wait 10s, then disconnect if the
        bot is still alone in its voice channel. Cancelled if a human rejoins."""
        try:
            mp = self.mps.get(guild.id)

            if mp is not None:
                try:
                    # send_with_np, not a bare channel send: this can fire mid-song
                    # and a bare send would bury the NP host message.
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

            # Span covers only the post-sleep decision, so it isn't open for the
            # full 10s (which confuses OTLP exporters and leaks OTel context).
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
        """Fires on cold start or session loss (not on WebSocket resume).
        Spawns a recovery task per guild so we don't block the event loop."""
        if self.redis is None:
            return
        # bot.guilds is empty until READY, so cog_load's pass covers an extension
        # reload and this one covers a cold start. Both are needed.
        spawn_background(self._load_debug_overrides(), self._restore_tasks)
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
        # Distributed lock so two bot instances can't race on the same guild.
        # Acquired inside the span so the SET NX EX is a child span.
        if not await store.acquire_recovery_lock():
            trace.get_current_span().set_attribute("restore.skipped_lock", True)
            log.info(
                f"Recovery lock held by another instance for guild {guild.id}, skipping"
            )
            return
        try:
            # One pipelined read serves both gates below: connection (state hash) and
            # anything-to-restore (queue length + crashed song). _restore_state
            # re-reads the real payload after a successful connect, so a stopped
            # guild's leftover queue never rides the wire on the nothing-to-do path.
            gate = await store.get_recovery_gate()
            if gate is None:
                # Read failed — do not treat as "nothing to restore". Skip this
                # attempt; the lock expires in 60s and the next on_ready retries.
                log.warning(f"Recovery skipped for guild {guild.id}: state read failed")
                return
            guild_state = gate.state
            # Equivalent to `not has_active_connection`, spelled as explicit None
            # checks so the channel IDs narrow to int below.
            vc_id = guild_state.voice_channel_id
            tc_id = guild_state.text_channel_id
            if vc_id is None or tc_id is None:
                return

            voice_channel = guild.get_channel(vc_id)
            text_channel = guild.get_channel(tc_id)
            voice_ok = isinstance(voice_channel, discord.VoiceChannel)
            text_ok = isinstance(text_channel, discord.TextChannel)

            if not voice_ok or not text_ok:
                # Clear stale IDs so this guild isn't re-attempted every reconnect.
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
        """Two cases: the bot itself disconnected/moved (full cleanup or
        stale-timer cancellation), and a human's channel change relative to the
        bot's (starts/cancels the 10s alone-disconnect countdown)."""
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
                # Bot moved — cancel any stale timer counting down the old channel.
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
