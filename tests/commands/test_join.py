"""Tests for `-join` (src/commands/join.py)."""

from unittest.mock import AsyncMock, MagicMock, patch

import discord
from redis.asyncio import Redis

from src.musicbot import MusicBot
from tests.helpers import (
    command_callback,
    described,
)


class TestJoinChannelPersistence:
    async def test_join_writes_channel_ids_to_redis(
        self,
        music_bot_with_redis: MusicBot,
        mock_ctx: MagicMock,
        mock_guild: MagicMock,
        fake_redis_bot: Redis,
    ) -> None:
        """Calling join should persist voice and text channel IDs to Redis."""
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 777000000000000001
        connected = MagicMock(spec=discord.VoiceClient)
        connected.is_connected.return_value = True
        connected.channel = voice_channel

        async def _connect(*_: object, **__: object) -> MagicMock:
            # What the real connect() does and a bare AsyncMock did not: leave a
            # CONNECTED client on the context. run() refuses to persist the channel
            # ids or open the gate without one.
            mock_ctx.voice_client = connected
            return connected

        voice_channel.connect = AsyncMock(side_effect=_connect)
        mock_ctx.author.voice.channel = voice_channel
        mock_guild.change_voice_state = AsyncMock()
        mock_guild.voice_client = None

        text_channel = MagicMock(spec=discord.TextChannel)
        text_channel.id = 777000000000000002
        mock_ctx.channel = text_channel

        mp = MagicMock()
        mp.store = MagicMock()
        mp.store.set_connection = AsyncMock()
        music_bot_with_redis.mps[mock_guild.id] = mp

        # join is a @commands.command — call the underlying callback directly.
        mock_ctx.voice_client = None  # bot not yet in channel
        with (
            patch.object(discord.VoiceChannel, "connect", new=AsyncMock()),
            patch.object(mock_ctx, "invoke", new=AsyncMock()),
        ):
            music_bot_with_redis.get_mp = MagicMock(return_value=mp)
            await command_callback(MusicBot.join)(music_bot_with_redis, mock_ctx)

        mp.store.set_connection.assert_awaited_once_with(
            voice_channel.id, text_channel.id
        )
        # Voice is up — a queue persisted by a previous -stop resumes.
        mp.open_playback_gate.assert_called_once()


class TestJoinReplacesAParkedClient:
    """discord.py registers the voice client BEFORE the handshake lands, and a
    cancelled or abandoned one is left registered. `ctx.voice_client` answers truthy
    for it, so a join that took it at its word would skip its own connect. -join is
    what an operator reaches for when the bot looks connected and is not, so it
    unregisters the stale client and connects a fresh one."""

    @staticmethod
    def _parked(
        mock_ctx: MagicMock, mock_guild: MagicMock, *, handshake_lands: bool = True
    ) -> tuple[MagicMock, MagicMock]:
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 777000000000000003
        mock_ctx.author.voice.channel = voice_channel
        mock_guild.change_voice_state = AsyncMock()

        parked = MagicMock(spec=discord.VoiceClient)
        parked.is_connected.return_value = False  # registered, handshake never landed
        parked.channel = voice_channel

        def _unregister(*_a: object, **_k: object) -> None:
            mock_ctx.voice_client = None
            mock_guild.voice_client = None

        parked.disconnect = AsyncMock(side_effect=_unregister)
        mock_ctx.voice_client = parked
        mock_guild.voice_client = parked

        def _connect(*_a: object, **_k: object) -> None:
            fresh = MagicMock(spec=discord.VoiceClient)
            fresh.is_connected.return_value = handshake_lands
            fresh.channel = voice_channel
            mock_ctx.voice_client = fresh
            mock_guild.voice_client = fresh

        voice_channel.connect = AsyncMock(side_effect=_connect)
        return voice_channel, parked

    @staticmethod
    def _player(music_bot: MusicBot) -> MagicMock:
        mp = MagicMock()
        mp.store = MagicMock()
        mp.store.set_connection = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)
        return mp

    async def test_one_join_replaces_it_and_connects(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        voice_channel, parked = self._parked(mock_ctx, mock_guild)
        mock_guild.me.voice = None  # Discord does not have the bot anywhere
        mp = self._player(music_bot)

        await command_callback(MusicBot.join)(music_bot, mock_ctx)

        # Unforced: below `connected` that is a local unregister, no op-4 wait.
        parked.disconnect.assert_awaited_once_with(force=False)
        voice_channel.connect.assert_awaited_once()
        mp.open_playback_gate.assert_called_once()
        mp.store.set_connection.assert_awaited_once()

    async def test_it_clears_discords_side_when_discord_still_has_the_bot(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        voice_channel, parked = self._parked(mock_ctx, mock_guild)
        mock_guild.me.voice = MagicMock(channel=voice_channel)
        self._player(music_bot)

        await command_callback(MusicBot.join)(music_bot, mock_ctx)

        parked.disconnect.assert_awaited_once_with(force=True)

    async def test_a_connected_client_is_left_alone(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        voice_channel, parked = self._parked(mock_ctx, mock_guild)
        parked.is_connected.return_value = True
        mp = self._player(music_bot)

        await command_callback(MusicBot.join)(music_bot, mock_ctx)

        parked.disconnect.assert_not_awaited()
        voice_channel.connect.assert_not_awaited()
        mp.open_playback_gate.assert_called_once()

    async def test_a_fresh_join_that_never_lands_is_refused_and_saves_nothing(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        self._parked(mock_ctx, mock_guild, handshake_lands=False)
        mock_guild.me.voice = None
        mp = self._player(music_bot)

        await command_callback(MusicBot.join)(music_bot, mock_ctx)

        mp.open_playback_gate.assert_not_called()
        mp.store.set_connection.assert_not_awaited()
        mock_guild.change_voice_state.assert_not_awaited()
        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Couldn't finish connecting" in described(embed)
        assert embed.color == discord.Color.red()
