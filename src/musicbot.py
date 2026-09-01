import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import (
    Any,
    Optional,
    Union,
)
from collections.abc import AsyncGenerator, Awaitable, Callable

import discord
from discord.ext import commands

import redis.asyncio as aioredis

from src.config import (
    SPOTIFY_TEST_TRACK_ID,
    SpotifyStatus,
    spotify_enabled,
)
from src import debug as debug_mode
from src.commands import clear as clear_cmd
from src.commands import debug as debug_cmd
from src.commands import history as history_cmd
from src.commands import join as join_cmd
from src.commands import jump as jump_cmd
from src.commands import leaderboard as leaderboard_cmd
from src.commands import now as now_cmd
from src.commands import pause as pause_cmd
from src.commands import play as play_cmd
from src.commands import playnow as playnow_cmd
from src.commands import queue as queue_cmd
from src.commands import remove as remove_cmd
from src.commands import resume as resume_cmd
from src.commands import shuffle as shuffle_cmd
from src.commands import skip as skip_cmd
from src.commands import stop as stop_cmd
from src.commands import volume as volume_cmd
from src.commands.history import (
    HISTORY_MAX_LIMIT,
    HISTORY_MIN_LIMIT,
    HistoryFlags,
)
from src.play_pipeline import PlaylistInputError
from src import analytics_card
from src.analytics_card import AnalyticsFlags
from src.commands.leaderboard import LeaderboardFlags
from src.history_archive import (
    ArchiveReader,
)
from src.musicplayer import MusicPlayer
from src.spotify import (
    Spotify,
    SpotifyAuthError,
    SpotifyRateLimitError,
    SpotifyRequestError,
)
from src.youtube import ExtractionError
from contextvars import Token

from opentelemetry import context as otel_context
from opentelemetry.context import Context
from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

from src.ping import run_health_dashboard
from src.recovery import VoiceWatchdog, restore_guild
from src.telemetry import get_tracer
from src.util import (
    cancel_task,
    notice_embed,
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
                self.mps[ctx.guild.id].home_channel
                if ctx.guild.id in self.mps
                else None
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
        elif isinstance(error, commands.CommandOnCooldown):
            # Raised in prepare(), before the body. The retry seconds are named so
            # the refusal reads as a limit rather than a fault.
            await ctx.send(
                embed=notice_embed(
                    f"That was just run here — try again in {error.retry_after:.0f}s.",
                    discord.Color.orange(),
                )
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
            "`?t=` / `?ts=` timestamp starts the song at that offset.\n\n"
            "A YouTube playlist link copied from partway through — one carrying "
            "an `&index=` — queues from that position, skipping the songs "
            "before it. Drop the `&index=` to queue the whole playlist."
        ),
        extras={
            "category": "Playback",
            "examples": [
                "-play never gonna give you up",
                "-play https://youtu.be/dQw4w9WgXcQ?t=43",
                "-play https://www.youtube.com/playlist?list=PLabc&index=4",
                "-play https://open.spotify.com/playlist/37i9dQZF1DXcBWIGoYBM5M",
                "-p https://soundcloud.com/artist/track",
            ],
            "note": (
                "Spotify links are matched to YouTube audio one title at a time, "
                "so a long playlist takes a few seconds to finish queueing."
            ),
        },
    )
    # Serialized per guild, like -resume: two concurrent invocations both read a
    # live current_song and both park a resume tail for it, so one play comes back
    # twice. wait=False — the second caller is told to wait rather than queued
    # behind a 1-4s extraction.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.play")
    async def play(self, ctx: commands.Context, *, url: str) -> None:
        try:
            await play_cmd.run(ctx, url, cog=self)
        except Exception as e:
            await self._command_error(ctx, e, title="Failed to queue song")

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
                "will not return. Otherwise they stack: run it again and the song "
                "you just interrupted waits its turn too, each one resuming from "
                "where it left off, most recent first."
            ),
        },
    )
    # See -play: interject() re-checks current_song, but current_song outlives
    # the check by a whole song, so the re-check cannot serialize two callers.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.playnow")
    async def playnow(self, ctx: commands.Context, *, url: str) -> None:
        try:
            await playnow_cmd.run(ctx, url, cog=self)
        except Exception as e:
            await self._command_error(ctx, e, title="Failed to play song now")

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
            # validate_commands rejects DMs, so a guild is guaranteed. mps, not
            # get_mp: -skip must not build a player.
            assert ctx.guild is not None
            await skip_cmd.run(ctx, mp=self.mps.get(ctx.guild.id))
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
            await stop_cmd.run(ctx, cog=self)
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
            await pause_cmd.run(ctx, mp=self.get_mp(ctx))
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
            await resume_cmd.run(ctx, cog=self)
        except Exception as e:
            await self._command_error(ctx, e)

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
            await shuffle_cmd.run(ctx, mp=self.get_mp(ctx))
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
            await join_cmd.run(ctx, mp=self.get_mp(ctx), bot_latency=self.bot.latency)
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
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.clear")
    async def clear(self, ctx: commands.Context) -> None:
        try:
            await clear_cmd.run(ctx, mp=self.get_mp(ctx))
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
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @commands.before_invoke(validate_commands)
    @_tracer.start_as_current_span("bot.remove")
    async def remove(
        self, ctx: commands.Context, *, needle: Optional[str] = None
    ) -> None:
        try:
            await remove_cmd.run(ctx, needle, mp=self.get_mp(ctx))
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
            await now_cmd.run(ctx, mp=self.get_mp(ctx))
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
            await history_cmd.run(ctx, flags, history=self.get_mp(ctx).history)
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
            await leaderboard_cmd.run(
                ctx, flags, archive=self.history_archive, redis=self.redis
            )
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
        name="analytics",
        aliases=["an"],
        brief="charts and totals for this server (long-term archive)",
        usage="[--days N]",
        help=(
            "Shows a chart of this server's listening — plays per day by source, "
            "when the server listens, listening time, how much of each song gets "
            "played, song lengths and queue wait — with the top listeners, artists "
            "and songs beside it.\n\n"
            "`--days N` picks the window: 7, 30, 90 or 365. It covers COMPLETE "
            "days, so today is not included — `-history` shows what just played. "
            "The numbers come from this server's long-term play archive."
        ),
        extras={
            "category": "Queue",
            # Read by cog_before_invoke to skip get_mp(): this command reads the
            # archive and never touches voice, so manufacturing a player for it
            # starts _restore_state() and a 300s gate on a guild doing nothing.
            "observation_only": True,
            "examples": ["-analytics", "-an", "-analytics --days 90"],
            "note": (
                "Available only when this server's host has enabled the "
                "optional long-term archive. Times are UTC."
            ),
        },
    )
    # No validate_commands: reading a chart needs no voice channel. Two bounds on
    # different axes — max_concurrency bounds how many run at once, the cooldown how
    # often. See docs/ARCHITECTURE.md#analytics-rendering.
    @commands.max_concurrency(1, commands.BucketType.guild, wait=False)
    @commands.cooldown(1, 30.0, commands.BucketType.guild)
    @_tracer.start_as_current_span("bot.analytics")
    async def analytics(self, ctx: commands.Context, *, flags: AnalyticsFlags) -> None:
        try:
            await analytics_card.run(
                ctx,
                flags,
                archive=self.history_archive,
                redis=self.redis,
                tasks=self._restore_tasks,
            )
        except Exception as e:
            # Fixed copy, as -leaderboard does: the default detail would publish the
            # archive's host and port. The trace footer still joins the span.
            await self._command_error(
                ctx,
                e,
                title="Analytics unavailable",
                detail=(
                    "The long-term archive could not be reached. Try again in a moment."
                ),
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
            await jump_cmd.run(ctx)
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
            await queue_cmd.run(ctx, mp=self.get_mp(ctx))
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
            await volume_cmd.run(ctx, volume, mp=self.get_mp(ctx))
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
                debug_suffix=self.debug_suffix(ctx),
            )
        except Exception as e:
            await self._command_error(ctx, e)

    # ── Debug mode ────────────────────────────────────────────────────────────
    # The state machine is DebugSettings (src/debug.py); what stays here is the
    # command surface and the permission policy around the toggle.

    def debug_suffix(
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
            await debug_cmd.run(ctx, arg, cog=self)
        except Exception as e:
            await self._command_error(ctx, e)

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
