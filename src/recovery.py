"""Voice-session lifecycle: rejoining after a restart, and leaving when alone.

restore_guild() is the join side (crash recovery,
docs/ARCHITECTURE.md#crash-recovery); VoiceWatchdog is the leave side (the
alone-disconnect countdown, posted as a card that ticks down and then says which
way it went), and its bot-was-ejected arm routes straight to cog.cleanup(). The
two cold-start helpers at the bottom are what -play and -resume both do about a
join that produced no usable voice client; they live here so the two commands'
checks cannot diverge.

Both take the MusicBot cog as an explicit parameter, the way MusicPlayer does.

Do not rename the `guild.restore` span — Tempo queries match on it.
"""

import asyncio
import contextlib
import math
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import commands
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src.musicplayer import MusicPlayer
from src.redis_client import GuildRedisStore
from src.telemetry import get_tracer
from src.util import (
    first_sendable_channel,
    get_logger,
    notice_embed,
    record_span_error,
)

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module).
    from src.musicbot import MusicBot

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# How often the countdown card is re-rendered. One second sits at Discord's
# per-channel edit ceiling, and every frame derives its number from the deadline
# rather than decrementing a counter, so an edit the rate limiter paces late costs
# a frame instead of quoting a number the clock has already passed.
_COUNTDOWN_TICK_SECS = 1.0

# The countdown bar drains rather than fills. Same width and glyphs as the Now
# Playing bar, the only other bar a guild sees.
_COUNTDOWN_BAR_WIDTH = 10
_COUNTDOWN_BAR_LEFT = "🟦"
_COUNTDOWN_BAR_SPENT = "⬜"


def _countdown_bar(remaining_secs: float, total_secs: float) -> str:
    """A `total_secs` bar with `remaining_secs` still to run."""
    ratio = remaining_secs / total_secs if total_secs > 0 else 0.0
    left = round(max(0.0, min(ratio, 1.0)) * _COUNTDOWN_BAR_WIDTH)
    return _COUNTDOWN_BAR_LEFT * left + _COUNTDOWN_BAR_SPENT * (
        _COUNTDOWN_BAR_WIDTH - left
    )


def _seconds_left(deadline: float, now: float) -> int:
    """Whole seconds still on the clock, floored at zero. Every frame derives its
    number this way instead of decrementing, so a late edit skips a number rather
    than quoting one the clock has already passed."""
    return max(0, math.ceil(deadline - now))


def _countdown_embed(remaining_secs: int, total_secs: float) -> discord.Embed:
    """One frame of the live card, built from whole seconds left."""
    unit = "second" if remaining_secs == 1 else "seconds"
    return discord.Embed(
        title="No users remaining in voice channel",
        description=(
            f"All users have disconnected. The bot will disconnect in "
            f"**{remaining_secs} {unit}** unless someone rejoins.\n\n"
            f"{_countdown_bar(remaining_secs, total_secs)}"
        ),
        color=discord.Color.orange(),
    )


def _rejoined_embed() -> discord.Embed:
    """Final frame when the countdown ends because someone came back."""
    return discord.Embed(
        title="Someone rejoined",
        description="The disconnect countdown was cancelled — playback continues.",
        color=discord.Color.green(),
    )


def _disconnected_embed() -> discord.Embed:
    """Final frame when the countdown runs out. Deliberately does not claim to be
    the cause: the bot can also have left voice by another route while this ran."""
    return discord.Embed(
        title="Disconnected from voice channel",
        description=(
            "The bot has left the voice channel. Its queue is kept for 24 hours — "
            "`-resume` picks it back up."
        ),
        color=discord.Color.dark_grey(),
    )


def _decorate(
    cog: MusicBot,
    guild: discord.Guild,
    embed: discord.Embed,
    span: Optional[trace.Span],
) -> discord.Embed:
    """Debug mode's footer for the two embeds this module sends itself: the
    channels-deleted notice and the alone-countdown card. Neither has a player to
    decorate it, so the cog's settings are read directly — one seam, not two.

    The card passes the span captured at its first send and reuses it for every
    frame. Re-derived per tick the footer would carry a fresh trace id naming a
    request that is already over, which is why the Now Playing card carries none.
    See docs/ARCHITECTURE.md#debug-footer-seams.
    """
    cog.debug_settings.decorate([embed], guild, span=span)
    return embed


@_tracer.start_as_current_span("guild.restore")
async def restore_guild(cog: MusicBot, guild: discord.Guild) -> None:
    """Attempt to rejoin voice and restore queue for one guild after restart."""
    if cog.redis is None:
        return
    if guild.id in cog.mps:
        return

    store = GuildRedisStore(cog.redis, guild.id)

    trace.get_current_span().set_attribute("discord.guild_id", str(guild.id))
    # One restore per guild at a time: on_ready re-fires on any reconnect that
    # fails to RESUME, and mps[guild.id] is set only after the connect below.
    # See docs/ARCHITECTURE.md#distributed-recovery-lock
    if not await store.acquire_recovery_lock():
        trace.get_current_span().set_attribute("restore.skipped_lock", True)
        log.info(f"Recovery already in progress for guild {guild.id}, skipping")
        return
    try:
        # One pipelined read serves both gates: connection (state hash) and
        # anything-to-restore (queue length + crashed song). _restore_state
        # re-reads the payload after a successful connect.
        gate = await store.get_recovery_gate()
        if gate is None:
            # Read failed, not "nothing to restore": the lock expires on its
            # TTL and the next on_ready retries.
            log.warning(f"Recovery skipped for guild {guild.id}: state read failed")
            return
        guild_state = gate.state
        # Explicit None checks so the channel IDs narrow to int.
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

            notify_channel: Optional[discord.TextChannel] = (
                text_channel if text_ok else first_sendable_channel(guild)
            )

            if notify_channel is not None:
                deleted: list[str] = []
                if not voice_ok:
                    deleted.append("voice channel")
                if not text_ok:
                    deleted.append("text channel")
                what = " and ".join(deleted)
                verb = "was" if len(deleted) == 1 else "were"
                notice = notice_embed(
                    f"⚠️ I came back online but the {what} I was playing in "
                    f"{verb} deleted. Use `-play` in a voice channel to start fresh.",
                    discord.Color.orange(),
                )
                _decorate(cog, guild, notice, trace.get_current_span())
                try:
                    await notify_channel.send(embed=notice)
                except Exception as notify_err:
                    log.warning(
                        f"Failed to send channel-deleted notification for "
                        f"guild {guild.id}: {notify_err}"
                    )
            return

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
            trace.get_current_span().set_attribute("restore.voice_connect_failed", True)
            trace.get_current_span().record_exception(e)
            trace.get_current_span().set_status(
                StatusCode.ERROR, f"voice connect failed: {e}"
            )
            log.warning(f"Could not rejoin voice for guild {guild.id}: {e}")
            return

        mp = MusicPlayer(cog.bot, guild, text_channel, cog, redis=cog.redis)
        mp.start()
        cog.mps[guild.id] = mp

        log.info(
            f"Restored guild {guild.id} in #{text_channel.name} / {voice_channel.name}"
        )
    except Exception as e:
        record_span_error(trace.get_current_span(), e)
        log.error(f"restore_guild failed for guild {guild.id}: {e}", exc_info=True)
    finally:
        await store.release_recovery_lock()


class VoiceWatchdog:
    """Disconnects the bot once it is alone in a voice channel, counting the
    warning down in the text channel while it waits.

    Owns the per-guild timer tasks and the rejoin events that end them early. Three
    rules are the whole of this feature's correctness and belong with the state they
    protect: cancel before pop, only the current task clears its own dict entries,
    and a rejoin SIGNALS a countdown rather than cancelling it — a cancelled task
    cannot edit its card to say what happened, so cancellation is left to teardown,
    which wants no card update anyway.

    One instance per cog, built in MusicBot.__init__.
    """

    __slots__ = ("_cog", "_rejoins", "_timers")

    def __init__(self, cog: MusicBot) -> None:
        self._cog = cog
        self._timers: dict[int, asyncio.Task] = {}
        self._rejoins: dict[int, asyncio.Event] = {}

    def cancel(self, guild_id: int) -> None:
        """Drop a guild's pending countdown, if any. Teardown only — see the class
        docstring for why a rejoin goes through _signal_rejoin instead.

        Never cancels the CALLING task: _countdown ends in cog.cleanup(), which
        calls straight back here, and cancelling yourself mid-teardown raises
        CancelledError out of cleanup and abandons the rest of it.
        """
        self._rejoins.pop(guild_id, None)
        existing = self._timers.pop(guild_id, None)
        if existing and not existing.done() and existing is not asyncio.current_task():
            existing.cancel()

    def _signal_rejoin(self, guild_id: int) -> None:
        """Someone is back — let the countdown run to its own end so it can finalize
        its card. The post-wait membership re-check is what actually stops the
        disconnect, so a countdown that never observes this event still cannot leave.
        Falls back to cancelling a timer that has no event to signal."""
        rejoined = self._rejoins.get(guild_id)
        if rejoined is None:
            self.cancel(guild_id)
            return
        log.info(f"User rejoined guild {guild_id}, ending alone countdown")
        rejoined.set()

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Two cases: the bot itself disconnected/moved (full cleanup or
        stale-timer cancellation), and a human's channel change relative to the
        bot's (starts/ends the alone-disconnect countdown)."""
        cog = self._cog
        guild = member.guild

        # ── Case A: bot itself was disconnected or moved ──────────────────────
        if cog.bot.user is not None and member.id == cog.bot.user.id:
            if before.channel is not None and after.channel is None:
                if guild.id in cog.mps:
                    with _tracer.start_as_current_span(
                        "bot.voice_state_update",
                        attributes={"discord.guild_id": str(guild.id)},
                    ):
                        log.info(
                            f"Bot disconnected from voice in guild {guild.id}, cleaning up"
                        )
                        await cog.cleanup(guild)
            elif before.channel is not None and after.channel is not None:
                # Moved — cancel any timer counting down the old channel.
                self.cancel(guild.id)
            return

        # ── Case B: a human member's voice state changed ──────────────────────
        if guild.id not in cog.mps:
            return

        vc = guild.voice_client
        if not isinstance(vc, discord.VoiceClient) or vc.channel is None:
            return

        # Mute/deafen events leave the channel unchanged.
        if before.channel == after.channel:
            return

        if before.channel != vc.channel and after.channel != vc.channel:
            return

        human_members = [m for m in vc.channel.members if not m.bot]

        if len(human_members) == 0:
            # Bot is now alone — start (or restart) the countdown. The event is
            # registered here rather than inside the task, so a human returning
            # before the coroutine's first line still ends it rather than falling
            # through to the cancel path and losing the card's final frame.
            self.cancel(guild.id)
            # Read here, synchronously: the card's frames, its deadline and the
            # log line all quote this one number.
            secs = cog.guild_settings.alone_timeout_secs(guild.id)
            log.info(
                f"Bot is alone in guild {guild.id}, starting {secs:g}s disconnect timer"
            )
            rejoined = asyncio.Event()
            self._rejoins[guild.id] = rejoined
            self._timers[guild.id] = asyncio.create_task(
                self._countdown(guild, secs, rejoined)
            )
        else:
            self._signal_rejoin(guild.id)

    async def _send_card(
        self,
        guild: discord.Guild,
        mp: MusicPlayer,
        embed: discord.Embed,
        span: Optional[trace.Span],
    ) -> Optional[discord.Message]:
        """Post the countdown card. A plain channel send, not send_with_np: this
        message is edited every tick, and a message an edit loop owns must never be
        the Now Playing host — the progress tick rebuilds a host from its cached
        send-time embeds and would undo every countdown frame. Same rule -ping and
        -debug follow, and _repin_now_playing pays back the burial it causes.

        None when the send fails; the countdown itself runs on regardless.
        """
        try:
            return await mp.home_channel.send(
                embed=_decorate(self._cog, guild, embed, span)
            )
        except Exception as e:
            log.warning(
                f"Failed to send alone-countdown notice in guild {guild.id}: {e}"
            )
            return None

    async def _push_card(
        self,
        guild: discord.Guild,
        message: discord.Message,
        embed: discord.Embed,
        span: Optional[trace.Span],
    ) -> Optional[discord.Message]:
        """Push one frame. None once the message is gone, so the loop stops spending
        edits on it; every other failure is swallowed and keeps the message, because
        a card that stopped updating must not stop the disconnect behind it."""
        try:
            await message.edit(embed=_decorate(self._cog, guild, embed, span))
            return message
        except discord.NotFound:
            return None
        except Exception as e:
            log.warning(f"alone-countdown card edit failed in guild {guild.id}: {e}")
            return message

    async def _repin_now_playing(self, guild: discord.Guild) -> None:
        """Re-host the Now Playing block at the channel bottom, where the card has
        been sitting on top of it. Only worth doing on the staying path — the
        leaving path retires the host inside cleanup()."""
        mp = self._cog.mps.get(guild.id)
        if mp is None:
            return
        with contextlib.suppress(Exception):
            await mp.repin_now_playing()

    async def _countdown(
        self, guild: discord.Guild, secs: float, rejoined: asyncio.Event
    ) -> None:
        """Post the countdown card, keep it ticking, then disconnect if the bot is
        still alone. Returns early — after a final frame — once `rejoined` is set.

        `secs` is this guild's own alone-timeout setting, read at the call, so the
        card opens on the number that guild chose rather than a fixed one.

        The deadline is fixed before the first send and every frame is derived from
        it, so a slow send or a paced edit costs a frame of the countdown rather
        than extending the wait behind it.
        """
        loop = asyncio.get_running_loop()
        deadline = loop.time() + secs
        message: Optional[discord.Message] = None
        card_span: Optional[trace.Span] = None
        try:
            mp = self._cog.mps.get(guild.id)
            if mp is not None:
                # Its own short span rather than one stretched over the countdown
                # (see below), held past the block so every later frame renders the
                # same footer.
                with _tracer.start_as_current_span(
                    "bot.alone_countdown.notice",
                    attributes={"discord.guild_id": str(guild.id)},
                ) as card_span:
                    opening = _seconds_left(deadline, loop.time())
                    message = await self._send_card(
                        guild, mp, _countdown_embed(opening, secs), card_span
                    )

            while True:
                remaining = deadline - loop.time()
                if remaining <= 0:
                    break
                with contextlib.suppress(TimeoutError):
                    async with asyncio.timeout(min(_COUNTDOWN_TICK_SECS, remaining)):
                        await rejoined.wait()
                if rejoined.is_set():
                    break
                left = _seconds_left(deadline, loop.time())
                # A frame reading zero is skipped: the final one lands on its heels
                # and says which way it went.
                if message is not None and left > 0:
                    message = await self._push_card(
                        guild, message, _countdown_embed(left, secs), card_span
                    )

            # Span covers only the post-countdown decision, so it isn't open for the
            # full wait (which confuses OTLP exporters and leaks OTel context).
            with _tracer.start_as_current_span(
                "bot.alone_countdown",
                attributes={"discord.guild_id": str(guild.id)},
            ):
                vc = guild.voice_client
                if not isinstance(vc, discord.VoiceClient) or vc.channel is None:
                    # The bot left voice by another route while this counted down;
                    # cleanup() has run or is about to, so don't run a second one.
                    await self._close_card(
                        guild, message, _disconnected_embed(), card_span
                    )
                elif rejoined.is_set() or any(not m.bot for m in vc.channel.members):
                    # The membership read is the authority; the event only makes the
                    # card prompt, and covers a rejoin the gateway never reported.
                    await self._close_card(guild, message, _rejoined_embed(), card_span)
                    await self._repin_now_playing(guild)
                else:
                    log.info(
                        f"Bot still alone in guild {guild.id} after "
                        f"{secs:g}s — disconnecting"
                    )
                    await self._close_card(
                        guild, message, _disconnected_embed(), card_span
                    )
                    await self._cog.cleanup(guild)
        except asyncio.CancelledError:
            pass  # explicit stop or teardown; the card is left where it stopped
        except Exception as e:
            log.error(f"alone countdown error in guild {guild.id}: {e}", exc_info=True)
        finally:
            # Only ever clear our OWN entries: a restart registers the replacement
            # before this task processes its cancellation, and an unguarded pop here
            # would drop the new countdown out of both dicts — leaving it running
            # and unreachable, so a later rejoin could not stop it.
            if self._timers.get(guild.id) is asyncio.current_task():
                del self._timers[guild.id]
            if self._rejoins.get(guild.id) is rejoined:
                del self._rejoins[guild.id]

    async def _close_card(
        self,
        guild: discord.Guild,
        message: Optional[discord.Message],
        embed: discord.Embed,
        span: Optional[trace.Span],
    ) -> None:
        """The countdown's last frame, if it ever got a card up."""
        if message is not None:
            await self._push_card(guild, message, embed, span)


def join_succeeded(ctx: commands.Context) -> bool:
    """Did the join a cold-start command just ran leave a USABLE voice client?
    is_connected(), not just the type: discord.py registers the client on the
    guild BEFORE the handshake completes, and vc.play() on a still-connecting
    one raises once per restored song. A failed join arrives as an absent
    client, not an exception."""
    vc = ctx.voice_client
    return isinstance(vc, discord.VoiceClient) and vc.is_connected()


async def abandon_cold_start(
    cog: MusicBot, ctx: commands.Context, mp: MusicPlayer
) -> None:
    """Drop the player a cold-start command (`-play`, `-resume`) was about to
    hand a voice connection to. defer_playback opens the gate as it unwinds
    whether or not the join worked, and a loop waking with no voice client
    fails its `vc` assertion once per restored song, draining the in-memory
    queue while Redis keeps every entry; tearing down first makes that
    gate-open land on a cancelled loop. The re-park FOLLOWS cleanup(), whose
    clear_connection() HDELs the fields it writes. Skipped while another
    command holds the gate: it is mid-join on this player and owns the
    teardown."""
    if mp.playback_holds > 1:  # this command's own hold, plus someone else's
        return
    if ctx.guild is not None:
        with contextlib.suppress(Exception):
            await cog.cleanup(ctx.guild)
    with contextlib.suppress(Exception):
        await mp.repark_crashed_head()
