"""Voice-session lifecycle: rejoining after a restart, and leaving when alone.

restore_guild() is the join side (crash recovery,
docs/ARCHITECTURE.md#crash-recovery); VoiceWatchdog is the leave side (the 10s
alone-disconnect countdown), and its bot-was-ejected arm routes straight to
cog.cleanup(). The two cold-start helpers at the bottom are what -play and
-resume both do about a join that produced no usable voice client; they live
here so the two commands' checks cannot diverge.

Both take the MusicBot cog as an explicit parameter, the way MusicPlayer does.

Do not rename the `guild.restore` span — Tempo queries match on it.
"""

import asyncio
import contextlib
from typing import TYPE_CHECKING, Optional

import discord
from discord.ext import commands
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src import config
from src.musicplayer import MusicPlayer
from src.redis_client import GuildRedisStore
from src.telemetry import get_tracer
from src.util import first_sendable_channel, get_logger, notice_embed, record_span_error

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module).
    from src.musicbot import MusicBot

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# How long the bot waits alone in a voice channel before disconnecting, and the
# number the countdown notice quotes.
ALONE_DISCONNECT_SECS = 10


def recovery_limiter() -> asyncio.Semaphore:
    """One per on_ready: admits RECOVERY_CONCURRENCY restores at a time, so the
    fan-out's Redis draw and voice connects are bounded by the knob rather than
    by the guild count. See docs/ARCHITECTURE.md#crash-recovery."""
    return asyncio.Semaphore(config.RECOVERY_CONCURRENCY)


async def restore_guild_bounded(
    cog: MusicBot, guild: discord.Guild, limiter: asyncio.Semaphore
) -> None:
    """restore_guild once the limiter admits it. The wait sits outside the
    `guild.restore` span, so a span's duration is the restore, not the queue."""
    async with limiter:
        await restore_guild(cog, guild)


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
    # None is a Redis failure, not a holder: the lock was never taken, so there
    # is nothing to release, and the next on_ready retries.
    # See docs/ARCHITECTURE.md#distributed-recovery-lock
    acquired = await store.acquire_recovery_lock()
    if acquired is None:
        trace.get_current_span().set_attribute("restore.lock_failed", True)
        log.warning(
            f"Recovery skipped for guild {guild.id}: recovery lock could not be "
            f"acquired (Redis error); retried on the next on_ready"
        )
        return
    if not acquired:
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
                # No player exists on this path, so the cog decorates directly.
                cog.debug_settings.decorate(
                    [notice], guild, span=trace.get_current_span()
                )
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
            # self_deaf rides the connect's own voice-state update, so the
            # handshake is one gateway exchange.
            await voice_channel.connect(timeout=30.0, reconnect=True, self_deaf=True)
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
    """Disconnects the bot once it is alone in a voice channel. Owns the
    per-guild timer tasks; one instance per cog, built in MusicBot.__init__."""

    __slots__ = ("_cog", "_timers")

    def __init__(self, cog: MusicBot) -> None:
        self._cog = cog
        self._timers: dict[int, asyncio.Task] = {}

    def cancel(self, guild_id: int) -> None:
        """Drop a guild's pending countdown, if any. Never cancels the CALLING
        task: _countdown ends in cog.cleanup(), which calls back here, and
        cancelling yourself mid-teardown abandons the rest of cleanup."""
        existing = self._timers.pop(guild_id, None)
        if existing and not existing.done() and existing is not asyncio.current_task():
            existing.cancel()

    async def on_voice_state_update(
        self,
        member: discord.Member,
        before: discord.VoiceState,
        after: discord.VoiceState,
    ) -> None:
        """Two cases: the bot itself disconnected/moved (full cleanup or
        stale-timer cancellation), and a human's channel change relative to the
        bot's (starts/cancels the alone-disconnect countdown)."""
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
            self.cancel(guild.id)
            log.info(
                f"Bot is alone in guild {guild.id}, starting "
                f"{ALONE_DISCONNECT_SECS}s disconnect timer"
            )
            self._timers[guild.id] = asyncio.create_task(self._countdown(guild))
        else:
            if guild.id in self._timers:
                log.info(f"User rejoined guild {guild.id}, cancelling alone timer")
            self.cancel(guild.id)

    async def _countdown(self, guild: discord.Guild) -> None:
        """Warn the guild's text channel, wait, then disconnect if the bot is
        still alone. Cancelled if a human rejoins."""
        try:
            mp = self._cog.mps.get(guild.id)

            if mp is not None:
                # Its own short span: without one current, the notice's debug
                # footer carries no trace id.
                with _tracer.start_as_current_span(
                    "bot.alone_countdown.notice",
                    attributes={"discord.guild_id": str(guild.id)},
                ):
                    try:
                        # send_with_np: this can fire mid-song, and a bare send
                        # would bury the NP host message.
                        embed = discord.Embed(
                            title="No users remaining in voice channel",
                            description=(
                                f"All users have disconnected. The bot will "
                                f"disconnect in **{ALONE_DISCONNECT_SECS} seconds** "
                                f"unless someone rejoins."
                            ),
                            color=discord.Color.orange(),
                        )
                        await mp.send_with_np(embed=embed)
                    except Exception as e:
                        log.warning(
                            f"Failed to send alone-countdown notice in guild {guild.id}: {e}"
                        )

            await asyncio.sleep(ALONE_DISCONNECT_SECS)

            # Covers only the post-sleep decision: a span held open across the
            # countdown confuses OTLP exporters and leaks OTel context.
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
                        f"Bot still alone in guild {guild.id} after "
                        f"{ALONE_DISCONNECT_SECS}s — disconnecting"
                    )
                    await self._cog.cleanup(guild)
        except asyncio.CancelledError:
            pass  # user rejoined or explicit stop; timer was cancelled
        except Exception as e:
            log.error(f"alone countdown error in guild {guild.id}: {e}", exc_info=True)
        finally:
            self._timers.pop(guild.id, None)


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
