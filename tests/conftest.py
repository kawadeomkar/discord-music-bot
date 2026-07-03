"""Shared fixtures for the discord-music-bot test suite."""

from typing import Any, AsyncIterator, Callable, Iterator, Optional
from unittest.mock import AsyncMock, MagicMock

import discord
import fakeredis
import pytest
import structlog
from redis.asyncio import Redis

from src.musicplayer import MusicPlayer
from src.spotify import Spotify
from tests.helpers import noop_ffmpeg_init


@pytest.fixture(autouse=True, scope="session")
def configure_structlog_for_tests() -> None:
    """Configure structlog with minimal output for tests.

    Replaces the JSON renderer with a plain renderer so test output is readable,
    and drops the OTel context processor (no TracerProvider in tests).
    """
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=False,
    )


@pytest.fixture(autouse=True)
def reset_structlog_contextvars() -> Iterator[None]:
    """Clear structlog context variables between every test.

    Without this, a test that calls bind_contextvars(guild_id=...) would
    leak that context into subsequent tests via the ContextVar storage.
    """
    structlog.contextvars.clear_contextvars()
    yield
    structlog.contextvars.clear_contextvars()


@pytest.fixture
def mock_guild() -> MagicMock:
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
def mock_author() -> MagicMock:
    member = MagicMock(spec=discord.Member)
    member.id = 222222222222222222
    member.name = "testuser"
    member.mention = "<@222222222222222222>"
    member.voice = MagicMock()
    member.voice.channel = MagicMock(spec=discord.VoiceChannel)
    return member


@pytest.fixture
def mock_channel() -> MagicMock:
    channel = MagicMock(spec=discord.TextChannel)
    channel.send = AsyncMock()
    return channel


@pytest.fixture
def mock_message(mock_author: MagicMock, mock_channel: MagicMock) -> MagicMock:
    message = MagicMock(spec=discord.Message)
    message.author = mock_author
    message.channel = mock_channel
    message.content = "-play test song"
    message.add_reaction = AsyncMock()
    return message


@pytest.fixture
def mock_ctx(
    mock_guild: MagicMock,
    mock_author: MagicMock,
    mock_channel: MagicMock,
    mock_message: MagicMock,
) -> MagicMock:
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
def mock_bot(mock_guild: MagicMock) -> MagicMock:
    bot = MagicMock()
    bot.guilds = [mock_guild]
    bot.latency = 0.05
    bot.is_closed.return_value = False
    bot.wait_until_ready = AsyncMock()
    # No create_task mock needed — MusicPlayer.start() is never called in tests
    return bot


@pytest.fixture
async def fake_redis() -> AsyncIterator[Redis]:
    """In-memory Redis for tests. Async fixture so aclose() runs at teardown."""
    server = fakeredis.FakeServer()
    client = fakeredis.aioredis.FakeRedis(server=server, decode_responses=False)
    yield client
    await client.aclose()


@pytest.fixture
def music_player(
    mock_bot: MagicMock,
    mock_guild: MagicMock,
    mock_channel: MagicMock,
    mock_ctx: MagicMock,
    fake_redis: Redis,
) -> MusicPlayer:
    """Construct MusicPlayer with fake Redis. start() is NOT called — tests operate on state directly.

    loop() blocks on _restored until _restore_state() finishes (see its docstring
    for why); since start() never runs here, nothing would set it. Tests that
    exercise that race explicitly should clear it again before calling loop().
    """
    mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis)
    mp._restored.set()
    return mp


@pytest.fixture
def spotify(fake_redis: Redis) -> Spotify:
    """Spotify instance with fake Redis cache and no blocking auth call at construction."""
    from unittest.mock import patch

    with patch.dict(
        "os.environ",
        {"SPOTIFY_CLIENT_ID": "test_id", "SPOTIFY_CLIENT_SECRET": "test_secret"},
    ):
        return Spotify(redis=fake_redis)


@pytest.fixture
def ytdl_instance(
    mock_channel: MagicMock, mock_author: MagicMock
) -> Callable[..., Any]:
    """Factory that creates a YTDL instance with FFmpegOpusAudio.__init__ patched out."""
    from unittest.mock import patch
    import discord as d
    from src.youtube import YTDL

    def _make(data: Optional[dict] = None) -> Any:
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
        with patch.object(d.FFmpegOpusAudio, "__init__", new=noop_ffmpeg_init):
            return YTDL(
                mock_channel,
                default_data["url"],
                data=default_data,
                requester=mock_author,
            )

    return _make
