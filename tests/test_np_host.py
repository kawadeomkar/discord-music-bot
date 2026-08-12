"""Tests for src/np_host.py — the Now Playing host pointer and card disposal.

Drives NpHost directly rather than through MusicPlayer: the swap protocol, the
edit lock's strip/delete asymmetry and the by-id delete's guards are this
module's contract, and a player is not needed to state any of them. The adopt
GATE (is the block still current?) stays in test_musicplayer.py with the player
that owns it.
"""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import discord
import pytest

from src.np_host import NpHost, NpHostRef
from tests.helpers import mocked


@pytest.fixture
def spawned() -> set[asyncio.Task]:
    """Stands in for MusicPlayer._background_tasks. An adopt retires the message
    it displaces fire-and-forget, so a test gathers this to await the retirement."""
    return set()


@pytest.fixture
def np_host(mock_bot: MagicMock, mock_guild: MagicMock, spawned: set) -> NpHost:
    def _spawn(coro: Any) -> asyncio.Task:
        task = asyncio.ensure_future(coro)
        spawned.add(task)
        task.add_done_callback(spawned.discard)
        return task

    return NpHost(mock_bot, mock_guild, _spawn)


class TestNpHostAdoptRetire:
    def test_adopt_updates_state_synchronously(
        self, np_host: NpHost, spawned: set
    ) -> None:
        msg = MagicMock(spec=discord.Message)
        msg.id = 1
        own = [discord.Embed(title="Queue")]
        np_host.adopt(msg, own)
        assert np_host.message is msg
        assert np_host.own_embeds is own
        assert np_host.dedicated is False
        assert not spawned  # no old host → no retire

    async def test_adopt_retires_old_dedicated_host_with_delete(
        self, np_host: NpHost, spawned: set
    ) -> None:
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        np_host.adopt(old, [], dedicated=True)
        new = AsyncMock(spec=discord.Message)
        new.id = 2
        np_host.adopt(new, [])
        await asyncio.gather(*list(spawned))
        old.delete.assert_awaited_once()
        old.edit.assert_not_awaited()

    async def test_adopt_strips_old_response_host_with_edit(
        self, np_host: NpHost, spawned: set
    ) -> None:
        old = AsyncMock(spec=discord.Message)
        old.id = 1
        old_own = [discord.Embed(title="Queue")]
        np_host.adopt(old, old_own)
        new = AsyncMock(spec=discord.Message)
        new.id = 2
        np_host.adopt(new, [], dedicated=True)
        await asyncio.gather(*list(spawned))
        old.edit.assert_awaited_once_with(embeds=old_own)
        old.delete.assert_not_awaited()

    async def test_adopt_same_message_retires_nothing(
        self, np_host: NpHost, spawned: set
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        msg.id = 1
        np_host.adopt(msg, [])
        np_host.adopt(msg, [discord.Embed(title="p")])
        assert not spawned
        msg.delete.assert_not_awaited()
        msg.edit.assert_not_awaited()

    async def test_retire_swallows_not_found(
        self, np_host: NpHost, spawned: set
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        msg.delete.side_effect = discord.NotFound(MagicMock(), "gone")
        await np_host.retire(msg, [], True)  # must not raise

    async def test_retire_swallows_and_logs_http_exception(
        self, np_host: NpHost, spawned: set
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        msg.edit.side_effect = discord.HTTPException(MagicMock(), "rate limited")
        await np_host.retire(msg, [], False)  # must not raise

    def test_release_clears_state_without_touching_message(
        self, np_host: NpHost, spawned: set
    ) -> None:
        msg = AsyncMock(spec=discord.Message)
        np_host._message = msg
        np_host._own_embeds = [discord.Embed(title="p")]
        np_host._dedicated = True
        np_host.release()
        assert np_host.message is None
        assert np_host.own_embeds == []
        assert np_host.dedicated is False
        msg.delete.assert_not_awaited()
        msg.edit.assert_not_awaited()

    async def test_adopt_ignores_older_message_and_sheds_its_block(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """Two overlapping sends can return out of order (channel position is
        send-start order, adopts run in send-return order) — an older message
        adopting late would pull the block up from the true bottom. The adopt
        is ignored and the older message sheds the block it carries."""
        newer = AsyncMock(spec=discord.Message)
        newer.id = 2
        np_host.adopt(newer, [])
        older = AsyncMock(spec=discord.Message)
        older.id = 1
        older_own = [discord.Embed(title="Queue")]
        np_host.adopt(older, older_own)
        await asyncio.gather(*list(spawned))
        assert np_host.message is newer
        older.edit.assert_awaited_once_with(embeds=older_own)
        newer.edit.assert_not_awaited()
        newer.delete.assert_not_awaited()

    async def test_retire_waits_for_lock_holder(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """Lock ordering on the STRIP: an in-flight tick edit (which holds the
        lock across its await) always completes before the strip, so the strip is
        the final write and a late tick cannot resurrect the NP block on the old
        host."""
        order: list[str] = []
        old = AsyncMock(spec=discord.Message)

        async def _edit(**_kw: Any) -> None:
            order.append("retire")

        old.edit.side_effect = _edit

        async def _hold_lock_like_a_tick() -> None:
            async with np_host.edit_lock:
                order.append("edit_started")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                order.append("edit_finished")

        holder = asyncio.create_task(_hold_lock_like_a_tick())
        await asyncio.sleep(0)  # holder acquires the lock
        retire = asyncio.create_task(np_host.retire(old, [], False))
        await asyncio.gather(holder, retire)
        assert order == ["edit_started", "edit_finished", "retire"]

    async def test_a_delete_does_not_queue_behind_the_edit_lock(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """And the asymmetry is deliberate. Nothing can resurrect a DELETED
        message — a late tick edit 404s and is swallowed — while message deletion
        is its own, stricter ratelimit bucket. Held across it, one 429 stalled
        every NP edit for the NEW song, so a burst of -playnow serialized the live
        progress bar behind a queue of deletes."""
        order: list[str] = []
        old = AsyncMock(spec=discord.Message)

        async def _delete() -> None:
            order.append("retire")

        old.delete.side_effect = _delete

        async def _hold_lock_like_a_tick() -> None:
            async with np_host.edit_lock:
                order.append("edit_started")
                await asyncio.sleep(0)
                await asyncio.sleep(0)
                order.append("edit_finished")

        holder = asyncio.create_task(_hold_lock_like_a_tick())
        await asyncio.sleep(0)  # holder acquires the lock
        retire = asyncio.create_task(np_host.retire(old, [], True))
        await asyncio.gather(holder, retire)

        assert order.index("retire") < order.index("edit_finished")


class TestRetireNpHostOnStop:
    """-stop / alone-disconnect teardown: the host is
    disposed of — unlike song end, which releases and leaves the completed bar
    as history, a bar frozen mid-song on a stopped player is misleading."""

    async def test_deletes_dedicated_host(self, np_host: NpHost, spawned: set) -> None:
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        np_host.adopt(host, [], dedicated=True)
        await np_host.retire_current()
        host.delete.assert_awaited_once()
        assert np_host.message is None

    async def test_strips_response_host_to_own_embeds(
        self, np_host: NpHost, spawned: set
    ) -> None:
        host = AsyncMock(spec=discord.Message)
        host.id = 1
        own = [discord.Embed(title="Queue")]
        np_host.adopt(host, own)
        await np_host.retire_current()
        host.edit.assert_awaited_once_with(embeds=own)
        host.delete.assert_not_awaited()
        assert np_host.message is None

    async def test_noop_when_no_host(self, np_host: NpHost, spawned: set) -> None:
        await np_host.retire_current()  # must not raise


class TestDisposePreviousNpCard:
    """Cleanup of the card an interrupted fragment left frozen. Without it a
    -playnow stack accumulates one dead partial bar per interjection: song end
    RELEASES the host rather than retiring it, by design."""

    def _song(self, **attrs: Any) -> MagicMock:
        song = MagicMock()
        song.np_message_id = 0
        song.np_channel_id = 0
        song.np_dedicated = False
        song.np_host_ref = None
        for name, value in attrs.items():
            setattr(song, name, value)
        return song

    async def test_dedicated_ref_is_deleted(
        self, np_host: NpHost, spawned: set
    ) -> None:
        message = AsyncMock(spec=discord.Message)
        song = self._song(np_host_ref=NpHostRef(message, [], True))

        await np_host.dispose_previous(song)

        message.delete.assert_awaited_once()
        message.edit.assert_not_awaited()

    async def test_a_channel_this_guild_does_not_own_is_never_touched(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """The ids are wire values up to the queue key's 24h TTL old, and a
        PartialMessageable validates nothing — so without scoping, a stale or
        corrupted channel id issues a DELETE wherever it resolves, including
        another guild or a DM."""
        mocked(np_host._guild).get_channel_or_thread = MagicMock(return_value=None)
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated=True,
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
        )

        await np_host.dispose_previous(song)

        np_host._bot.get_partial_messageable.assert_not_called()

    async def test_a_bool_id_never_reaches_the_route(
        self, np_host: NpHost, spawned: set
    ) -> None:
        # isinstance(True, int) is True in Python, so a wire `true` would render
        # "True" into the REST path. The isinstance check alone did not catch it.
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated=True, np_message_id=True, np_channel_id=888888888888888888
        )

        await np_host.dispose_previous(song)

        np_host._bot.get_partial_messageable.assert_not_called()

    async def test_a_half_stamped_pair_is_never_issued(
        self, np_host: NpHost, spawned: set
    ) -> None:
        # Both ids come off one message, so a zero on either side means the pair
        # never identified anything — get_partial_message(0) would 404 at best.
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated=True, np_message_id=777777777777777777, np_channel_id=0
        )

        await np_host.dispose_previous(song)

        np_host._bot.get_partial_messageable.assert_not_called()

    async def test_forbidden_and_http_errors_are_swallowed(
        self, np_host: NpHost, spawned: set
    ) -> None:
        # Fire-and-forget: a permission change after the card was posted is the
        # ordinary case and must not surface as an unretrieved task exception.
        for exc in (
            discord.Forbidden(MagicMock(status=403), "nope"),
            discord.HTTPException(MagicMock(status=500), "boom"),
            TimeoutError("aiohttp gave up"),
        ):
            message = AsyncMock(spec=discord.Message)
            message.delete = AsyncMock(side_effect=exc)
            song = self._song(np_host_ref=None, np_dedicated=True)
            song.np_message_id = 777777777777777777
            song.np_channel_id = 888888888888888888
            np_host._bot.get_partial_messageable = MagicMock(
                return_value=MagicMock(
                    get_partial_message=MagicMock(return_value=message)
                )
            )

            await np_host.dispose_previous(song)  # must not raise

    async def test_a_truthy_non_bool_dedicated_flag_never_authorizes_a_delete(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """np_dedicated is the AUTHORIZATION, not a target — the only thing
        between deleting the bot's own card and deleting a user's command reply.
        parse_queue_entry coerces nothing, so a wire "false" arrives as a truthy
        string; truthiness would read that as permission to delete."""
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_dedicated="false",  # truthy string, e.g. a "1"/"0" writer
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
        )

        await np_host.dispose_previous(song)

        np_host._bot.get_partial_messageable.assert_not_called()

    async def test_response_ref_is_strip_edited_back_to_its_own_embeds(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """A card hosted by a command response must NOT be deleted — that would
        destroy a user's reply. Only the live ref can do this: own_embeds cannot be
        reconstructed from ids, which is why the by-id path skips non-dedicated."""
        message = AsyncMock(spec=discord.Message)
        own = [discord.Embed(title="the reply's own embed")]
        song = self._song(np_host_ref=NpHostRef(message, own, False))

        await np_host.dispose_previous(song)

        message.edit.assert_awaited_once_with(embeds=own)
        message.delete.assert_not_awaited()

    async def test_wire_ids_delete_a_dedicated_card_by_id(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """The post-restart path: the ref is gone, the ids survived. No fetch and
        no cache lookup — a partial message issues the DELETE directly."""
        partial = MagicMock()
        partial.delete = AsyncMock()
        messageable = MagicMock()
        messageable.get_partial_message = MagicMock(return_value=partial)
        np_host._bot.get_partial_messageable = MagicMock(return_value=messageable)
        song = self._song(
            np_message_id=777777777777777777,
            np_channel_id=888888888888888888,
            np_dedicated=True,
        )

        await np_host.dispose_previous(song)

        # guild_id scopes the route: the ids are wire values up to 24h stale and
        # a PartialMessageable validates nothing on its own.
        mocked(np_host._bot.get_partial_messageable).assert_called_once_with(
            888888888888888888, guild_id=np_host._guild.id
        )
        messageable.get_partial_message.assert_called_once_with(777777777777777777)
        partial.delete.assert_awaited_once()

    async def test_by_id_delete_swallows_not_found(
        self, np_host: NpHost, spawned: set
    ) -> None:
        partial = MagicMock()
        partial.delete = AsyncMock(
            side_effect=discord.NotFound(MagicMock(status=404), "gone")
        )
        messageable = MagicMock()
        messageable.get_partial_message = MagicMock(return_value=partial)
        np_host._bot.get_partial_messageable = MagicMock(return_value=messageable)
        song = self._song(np_message_id=7, np_channel_id=8, np_dedicated=True)

        await np_host.dispose_previous(song)  # must not raise

        partial.delete.assert_awaited_once()

    async def test_wire_ids_alone_never_touch_a_response_host(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """Deliberately a no-op: a by-id DELETE of a non-dedicated host destroys a
        user's command reply, and a strip-edit needs embeds the ids cannot supply.
        A frozen block left on a response after a crash is accepted noise."""
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(np_message_id=7, np_channel_id=8, np_dedicated=False)

        await np_host.dispose_previous(song)

        mocked(np_host._bot.get_partial_messageable).assert_not_called()

    async def test_unstamped_song_is_a_noop(
        self, np_host: NpHost, spawned: set
    ) -> None:
        np_host._bot.get_partial_messageable = MagicMock()
        await np_host.dispose_previous(self._song())
        mocked(np_host._bot.get_partial_messageable).assert_not_called()

    async def test_a_non_integer_id_never_reaches_the_delete(
        self, np_host: NpHost, spawned: set
    ) -> None:
        """parse_queue_entry coerces nothing, so a corrupt entry can carry a
        non-id here. The guard is at the destructive call rather than in the
        parser — dropping the whole song over a cosmetic field would be worse."""
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_message_id={"nested": "object"}, np_channel_id=8, np_dedicated=True
        )

        await np_host.dispose_previous(song)

        mocked(np_host._bot.get_partial_messageable).assert_not_called()

    async def test_the_ref_wins_over_the_ids(
        self, np_host: NpHost, spawned: set
    ) -> None:
        # Both present (no crash, ids stamped anyway): the ref path is strictly
        # better — it can strip-edit — and doing both would double-retire.
        message = AsyncMock(spec=discord.Message)
        np_host._bot.get_partial_messageable = MagicMock()
        song = self._song(
            np_host_ref=NpHostRef(message, [], True),
            np_message_id=7,
            np_channel_id=8,
            np_dedicated=True,
        )

        await np_host.dispose_previous(song)

        message.delete.assert_awaited_once()
        mocked(np_host._bot.get_partial_messageable).assert_not_called()
