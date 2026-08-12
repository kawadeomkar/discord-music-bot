"""The Now Playing host — which message currently carries the live progress bar.

MusicPlayer builds the block and edits it; this module owns only the pointer to
the message showing it, the swap protocol that keeps that pointer at the channel
bottom, and the disposal of cards a swap or a -playnow interjection leaves
behind. Three rules make a swap safe and they live together here rather than
interleaved with playback: pointer-first adoption, the edit lock's strip/delete
asymmetry, and the hostile-input gate on a by-id delete.

The adopt GATE stays on MusicPlayer — whether the block still describes the
current song is playback's question, not this module's.

NpHostRef is here rather than beside QueueObject because a Discord card is not a
yt-dlp concept; youtube.py imports it back for the field a resume tail carries.

See docs/ARCHITECTURE.md#now-playing-host-model.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Optional, Protocol

import discord

from src.util import get_logger

log = get_logger(__name__)


def host_ids(message: Optional[discord.Message]) -> tuple[int, int]:
    """A host message as the (message_id, channel_id) pair everything at rest
    stores it as. Both come off one message or both are 0 — a card is only
    resolvable as a pair, and a channel id from any other source would 404 for
    exactly the host-migrated plays a pointer is wanted for. One function so that
    stays true of every writer rather than of each one separately."""
    if message is None:
        return 0, 0
    return message.id, message.channel.id


class NpCard(Protocol):
    """What carries a pointer to one Now Playing card: the live ref while the
    process that made it is alive, the wire ids always.

    A Protocol rather than a type, so this module states the SHAPE it needs and
    imports nothing to get it — QueueObject and YTDL satisfy it structurally, and
    a card is not a yt-dlp concept in either direction. The four fields move
    together at every site that touches them, which is why they are one parameter
    rather than four."""

    np_host_ref: Optional["NpHostRef"]
    np_message_id: int
    np_channel_id: int
    np_dedicated: bool


def stamp_card(
    target: NpCard,
    message: Optional[discord.Message],
    own_embeds: list[discord.Embed],
    dedicated: bool,
) -> None:
    """Record the host that was live when a fragment ended onto the tail that will
    dispose of it. The inverse of what NpHost.dispose_previous reads back, so the
    ref-or-ids pairing has one definition on the write side too."""
    target.np_host_ref = (
        NpHostRef(message, own_embeds, dedicated) if message is not None else None
    )
    target.np_message_id, target.np_channel_id = host_ids(message)
    target.np_dedicated = dedicated


@dataclass(frozen=True, slots=True)
class NpHostRef:
    """The live host an interrupted fragment left behind, so the fragment's resume
    tail can dispose of that frozen card when it starts.

    Runtime only — a live Message cannot be serialized, and own_embeds cannot be
    reconstructed from ids, so the wire fields alone can never strip-edit a
    retirement (see NpHost.retire)."""

    message: discord.Message
    own_embeds: list[discord.Embed]
    dedicated: bool


class NpHost:
    """The host pointer for one guild, plus the lock every edit against it takes.

    One instance per MusicPlayer. `spawn` is the player's fire-and-forget helper:
    an adopt retires the message it displaces without waiting for the HTTP call.
    """

    __slots__ = (
        "_bot",
        "_guild",
        "_spawn",
        "_message",
        "_own_embeds",
        "_dedicated",
        "_lock",
    )

    def __init__(
        self,
        bot: discord.Client,
        guild: discord.Guild,
        spawn: Callable[[Coroutine[Any, Any, Any]], asyncio.Task],
    ) -> None:
        self._bot = bot
        self._guild = guild
        self._spawn = spawn
        self._message: Optional[discord.Message] = None
        self._own_embeds: list[discord.Embed] = []
        self._dedicated: bool = False
        self._lock = asyncio.Lock()

    # ── Queries ───────────────────────────────────────────────────────────────

    @property
    def message(self) -> Optional[discord.Message]:
        """The message carrying the block, or None when the host is dormant."""
        return self._message

    @property
    def own_embeds(self) -> list[discord.Embed]:
        """The host's own embeds — what a strip-edit restores it to."""
        return self._own_embeds

    @property
    def dedicated(self) -> bool:
        """True for a pure NP message (deletable), False for a command response."""
        return self._dedicated

    @property
    def edit_lock(self) -> asyncio.Lock:
        """Held across every edit against the host. retire()'s strip takes it, so a
        caller's edit and a retirement cannot resolve last-write-wins server-side."""
        return self._lock

    def snapshot(self) -> tuple[Optional[discord.Message], list[discord.Embed], bool]:
        """All three fields as one tuple, for a caller that must capture the host
        before releasing it — the final bar edit is issued against this, not
        against whatever the next song adopts."""
        return self._message, self._own_embeds, self._dedicated

    # ── Transitions ───────────────────────────────────────────────────────────

    def adopt(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        *,
        dedicated: bool = False,
    ) -> None:
        """Pointer-first host swap. The pointer update is synchronous (atomic on the
        event loop), so any tick starting after this targets the new host. Retiring
        the old one is fire-and-forget; retire()'s lock orders it after any
        in-flight tick edit against that message."""
        old_msg = self._message
        old_own = self._own_embeds
        old_dedicated = self._dedicated
        if old_msg is not None and message.id < old_msg.id:
            # Overlapping sends can complete out of order: channel position is
            # send-START order, adopts run in send-RETURN order. Adopting the older
            # message would pull the block up from the true bottom — keep the newer
            # host and shed the older message's block instead.
            self.shed(message, own_embeds, dedicated)
            return
        self._message = message
        self._own_embeds = own_embeds
        self._dedicated = dedicated
        if old_msg is not None and old_msg.id != message.id:
            self.shed(old_msg, old_own, old_dedicated)

    def shed(
        self, message: discord.Message, own_embeds: list[discord.Embed], dedicated: bool
    ) -> None:
        """Retire a message that is not (or is no longer) the host, without
        waiting: the block it carries has nothing left to keep it current. Both of
        adopt's exits and the player's declined adopt gate all end here."""
        self._spawn(self.retire(message, own_embeds, dedicated))

    def release(self) -> None:
        """Clear host state without retiring the message. Used at song end: the
        completed bar stays in the channel as a record, and the next song's adopt
        sees no old host to retire."""
        self._message = None
        self._own_embeds = []
        self._dedicated = False

    def release_if(self, message: discord.Message) -> None:
        """Release only while `message` is still the host. Adopt is lock-free, so a
        command response can swap in a new host during an edit that was targeting
        the old one; releasing unconditionally would orphan the new host's block."""
        if self._message is message:
            self.release()

    async def retire(
        self,
        message: discord.Message,
        own_embeds: list[discord.Embed],
        dedicated: bool,
    ) -> None:
        """Remove the NP block from a message that is no longer the host.

        The STRIP takes the edit lock so an in-flight tick edit finishes first:
        concurrent PATCHes resolve last-write-wins server-side, and a tick landing
        after the strip would resurrect the block on the retired host.

        The DELETE does not. Nothing can resurrect a deleted message, while message
        deletion is its own stricter ratelimit bucket — held across it, one 429
        stalled every NP edit for the NEW song for the retry-after."""
        try:
            if dedicated:
                await message.delete()  # pure NP message → remove entirely
            else:
                async with self._lock:
                    # response → strip NP block, keep its own embeds
                    await message.edit(embeds=own_embeds)
        except discord.NotFound:
            pass  # user already deleted it — nothing to retire
        except discord.HTTPException as e:
            log.warning(f"NP host retire failed for guild {self._guild.id}: {e}")

    async def retire_current(self) -> None:
        """-stop / alone-disconnect teardown: dispose of the host so no message keeps
        a live-looking bar for a player that no longer exists. Song end RELEASES
        instead — a completed bar is a truthful record, a mid-song frozen one is not.
        cleanup() calls this after the progress/loop tasks are cancelled."""
        host, own, dedicated = self.snapshot()
        if host is None:
            return
        self.release()
        await self.retire(host, own, dedicated)

    async def dispose_previous(self, card: NpCard) -> None:
        """Remove the frozen card a previous fragment of one play left behind.

        Takes either form of the tail — the YTDL about to play, or the QueueObject a
        bulk mutation is destroying — since both satisfy NpCard. Without this a
        -playnow stack accumulates one dead partial bar per interjection.

        With the live ref this is retire() verbatim, so a card hosted by a command
        response is strip-edited back to its own embeds. After a restart only the
        ids survive and own_embeds cannot be reconstructed from them, so the by-id
        delete is gated to DEDICATED cards. Never a re-adopt: the live bar belongs
        at the channel bottom.
        """
        ref = card.np_host_ref
        if ref is not None:
            await self.retire(ref.message, ref.own_embeds, ref.dedicated)
            return
        message_id, channel_id = card.np_message_id, card.np_channel_id
        dedicated = card.np_dedicated
        # parse_queue_entry coerces nothing, so three wire values reach this
        # DESTRUCTIVE call unchecked. `dedicated` is the authorization between
        # "delete this message" and "leave a user's reply alone", so `is True` and
        # not truthiness — a wire "false" is a truthy string.
        if dedicated is not True:
            return
        # bool excluded: isinstance(True, int) is True, so a wire `true` would
        # render "True" into the REST route.
        if isinstance(message_id, bool) or isinstance(channel_id, bool):
            return
        if not (isinstance(message_id, int) and isinstance(channel_id, int)):
            return
        if not (message_id > 0 and channel_id > 0):
            return
        # Scoped to THIS guild first: a PartialMessageable validates nothing, so a
        # stale or corrupted channel id would delete a message wherever it happens
        # to resolve, including another guild or a DM.
        if self._guild.get_channel_or_thread(channel_id) is None:
            return
        # get_partial_messageable: issues the DELETE without the channel being
        # cached, which it may not be on the restart path that reaches this branch.
        channel = self._bot.get_partial_messageable(channel_id, guild_id=self._guild.id)
        try:
            await channel.get_partial_message(message_id).delete()
        except discord.NotFound:
            pass  # channel or message gone — nothing to clean up either way
        except discord.Forbidden:
            pass  # permissions changed since the card was posted
        except discord.HTTPException as e:
            log.warning(f"NP card cleanup failed for guild {self._guild.id}: {e}")
        except Exception as e:
            # discord.py surfaces aiohttp.ClientError and asyncio.TimeoutError once
            # its retries are spent, and this runs fire-and-forget — unhandled they
            # land outside structlog as "Task exception was never retrieved".
            log.warning(
                f"NP card cleanup errored for guild {self._guild.id}: "
                f"{type(e).__name__}: {e}"
            )
