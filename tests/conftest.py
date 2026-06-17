"""Shared fixtures for the discord-music-bot test suite."""

from unittest.mock import AsyncMock, MagicMock

import discord
import fakeredis
import pytest

from src.musicplayer import MusicPlayer
from src.spotify import Spotify


@pytest.fixture
def mock_guild():
    guild = MagicMock(spec=discord.Guild)
    guild.id = 111111111111111111
    guild.voice_client = MagicMock(spec=discord.VoiceClient)
    guild.voice_client.is_playing.return_value = False
    guild.voice_client.is_paused.return_value = False
    me = MagicMock(spec=discord.Member)
    me.id = 999999999999999999
    me.mention = "<@999999999999999999>"
    guild.me = me
    guild.owner = me
    return guild


@pytest.fixture
def mock_author():
    member = MagicMock(spec=discord.Member)
    member.id = 222222222222222222
    member.name = "testuser"
    member.mention = "<@222222222222222222>"
    member.voice = MagicMock()
    member.voice.channel = MagicMock(spec=discord.VoiceChannel)
    return member


@pytest.fixture
def mock_channel():
    channel = MagicMock(spec=discord.TextChannel)
    channel.send = AsyncMock()
    return channel


@pytest.fixture
def mock_message(mock_author, mock_channel):
    message = MagicMock(spec=discord.Message)
    message.author = mock_author
    message.channel = mock_channel
    message.content = "-play test song"
    message.add_reaction = AsyncMock()
    return message


@pytest.fixture
def mock_ctx(mock_guild, mock_author, mock_channel, mock_message):
    ctx = MagicMock()
    ctx.guild = mock_guild
    ctx.author = mock_author
    ctx.channel = mock_channel
    ctx.message = mock_message
    ctx.cog = MagicMock()
    ctx.send = AsyncMock()
    ctx.typing = MagicMock()
    ctx.typing.return_value.__aenter__ = AsyncMock(return_value=None)
    ctx.typing.return_value.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.fixture
def mock_bot(mock_guild):
    bot = MagicMock()
    bot.guilds = [mock_guild]
    bot.latency = 0.05
    bot.is_closed.return_value = False
    bot.wait_until_ready = AsyncMock()
    # No create_task mock needed — MusicPlayer.start() is never called in tests
    return bot


@pytest.fixture
async def fake_redis():
    """In-memory Redis for tests. Async fixture so aclose() runs at teardown."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def music_player(mock_bot, mock_guild, mock_channel, mock_ctx, fake_redis):
    """Construct MusicPlayer with fake Redis. start() is NOT called — tests operate on state directly."""
    return MusicPlayer(
        mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis
    )


@pytest.fixture
def spotify(fake_redis):
    """Spotify instance with fake Redis cache and no blocking auth call at construction."""
    from unittest.mock import patch

    with patch.dict(
        "os.environ",
        {"SPOTIFY_CLIENT_ID": "test_id", "SPOTIFY_CLIENT_SECRET": "test_secret"},
    ):
        return Spotify(redis=fake_redis)


@pytest.fixture
def ytdl_instance(mock_channel, mock_author):
    """Factory that creates a YTDL instance with FFmpegOpusAudio.__init__ patched out."""
    from unittest.mock import patch
    import discord as d
    from src.youtube import YTDL

    def _make(data=None):
        default_data = {
            "url": "https://r2.googlevideo.com/stream?expire=9999999999",
            "webpage_url": "https://www.youtube.com/watch?v=test",
            "title": "Test Song",
            "upload_date": "20240101",
            "duration": 180,
            "uploader": "Test Channel",
            "uploader_url": "",
            "thumbnail": "https://img.yt.com/test.jpg",
            "description": "",
            "tags": [],
            "view_count": 1000,
            "like_count": 100,
            "dislike_count": 5,
            "abr": 128,
            "asr": 44100,
            "acodec": "opus",
        }
        if data:
            default_data.update(data)
        with patch.object(d.FFmpegOpusAudio, "__init__", return_value=None):
            return YTDL(
                mock_channel,
                default_data["url"],
                data=default_data,
                requester=mock_author,
            )

    return _make
