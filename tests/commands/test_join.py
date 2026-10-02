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


class TestJoinRefusesAParkedClient:
    """discord.py registers the voice client BEFORE the handshake lands, and a
    cancelled or abandoned one is left registered. `ctx.voice_client` answers truthy
    for it, so run() skips its own connect and nothing raises — this is the only
    place that notices. Opening the gate there costs a song per loop iteration,
    draining the in-memory queue while Redis keeps every entry."""

    @staticmethod
    def _parked(mock_ctx: MagicMock, mock_guild: MagicMock) -> MagicMock:
        voice_channel = MagicMock(spec=discord.VoiceChannel)
        voice_channel.id = 777000000000000003
        voice_channel.connect = AsyncMock()
        mock_ctx.author.voice.channel = voice_channel
        mock_guild.change_voice_state = AsyncMock()

        parked = MagicMock(spec=discord.VoiceClient)
        parked.is_connected.return_value = False  # registered, handshake never landed
        parked.channel = voice_channel
        mock_ctx.voice_client = parked
        mock_guild.voice_client = parked
        return voice_channel

    async def test_the_gate_stays_shut_and_the_channel_is_not_persisted(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        voice_channel = self._parked(mock_ctx, mock_guild)
        mp = MagicMock()
        mp.store = MagicMock()
        mp.store.set_connection = AsyncMock()
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.join)(music_bot, mock_ctx)

        voice_channel.connect.assert_not_awaited()  # the parked client short-circuits it
        mp.open_playback_gate.assert_not_called()
        mp.store.set_connection.assert_not_awaited()
        mock_guild.change_voice_state.assert_not_awaited()

    async def test_the_author_is_told(
        self, music_bot: MusicBot, mock_ctx: MagicMock, mock_guild: MagicMock
    ) -> None:
        self._parked(mock_ctx, mock_guild)
        mp = MagicMock()
        mp.store = None
        music_bot.get_mp = MagicMock(return_value=mp)

        await command_callback(MusicBot.join)(music_bot, mock_ctx)

        embed = mock_ctx.send.await_args.kwargs["embed"]
        assert "Couldn't finish connecting" in described(embed)
        assert embed.color == discord.Color.red()
