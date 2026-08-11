"""Voice-session lifecycle: rejoining after a restart, and leaving when alone.

Two halves of one question — when does the bot join a voice channel, and when
does it leave one. restore_guild() is the join side (crash recovery, documented
in docs/ARCHITECTURE.md#crash-recovery); VoiceWatchdog is the leave side (the
10s alone-disconnect countdown). on_voice_state_update's bot-was-ejected arm is
the third case and routes straight to cog.cleanup().

These take the MusicBot cog as an explicit parameter rather than living on it.
They are functions ABOUT the cog, not commands, and naming the dependency is
what lets them out of a 2,400-line module. MusicPlayer already receives the cog
the same way.

The `guild.restore` span keeps its name, so existing Tempo queries still match;
what moved is the OTel instrumentation SCOPE, from src.musicbot to src.recovery.
"""

from typing import TYPE_CHECKING, Optional

import discord
from opentelemetry import trace
from opentelemetry.trace import StatusCode

from src import debug as debug_mode
from src.musicplayer import MusicPlayer
from src.redis_client import GuildRedisStore
from src.telemetry import get_tracer
from src.util import first_sendable_channel, get_logger, notice_embed, record_span_error

if TYPE_CHECKING:
    # A runtime import would close the cycle (musicbot imports this module); the
    # cog is only named in annotations. Same guard musicplayer.py and debug.py use.
    from src.musicbot import MusicBot

log = get_logger(__name__)
_tracer = get_tracer(__name__)

# How long the bot waits alone in a voice channel before disconnecting, and the
# number the countdown notice quotes. One constant so the two cannot disagree.
ALONE_DISCONNECT_SECS = 10


@_tracer.start_as_current_span("guild.restore")
async def restore_guild(cog: "MusicBot", guild: discord.Guild) -> None:
    """Attempt to rejoin voice and restore queue for one guild after restart."""
    if cog.redis is None:
        return
    if guild.id in cog.mps:
        return

    store = GuildRedisStore(cog.redis, guild.id)

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
                if cog.debug_enabled(guild.id):
                    debug_mode.decorate_embeds(
                        [notice],
                        span=trace.get_current_span(),
                        shard_id=guild.shard_id,
                        runtime=cog.runtime_snapshot,
                    )
                try:
                    await notify_channel.send(embed=notice)
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
