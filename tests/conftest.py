"""Shared fixtures for the discord-music-bot test suite."""

import re
import sys
from typing import Any, Optional, cast
from collections.abc import AsyncIterator, Callable, Iterator
from unittest.mock import AsyncMock, MagicMock

import discord
import fakeredis
import pytest
import structlog
from redis.asyncio import Redis

from src.config import SpotifyStatus
from src.musicbot import MusicBot
from src.musicplayer import MusicPlayer
from src.spotify import Spotify
from tests.helpers import noop_ffmpeg_init, tier_enabled


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Refuse a `-m pg` run that would skip the entire tier.

    An all-skipped tier is otherwise a GREEN job:

        $ env -u RUN_PG_TESTS -u POSTGRES_TEST_URL pytest -m pg -q
        25 skipped, 1465 deselected      EXIT=0

    `just test-pg` hardcodes RUN_PG_TESTS=1 so this cannot happen through
    workflow env alone, and CI's `build` needs: pg-integration — but nothing
    ASSERTED that any pg test actually ran, and several invariants are covered
    ONLY here: ON CONFLICT dedup, the -history tie-break, the schema lock in
    both directions (every constructible HistoryEntry inserts; a CHECK still
    refuses one that bypassed the validator), NOT VALID's treatment of legacy
    rows, and play_history_rejected.payload holding a NUL byte that jsonb and
    text both refuse. A tier that silently stops running is worse than one that
    was never wired up.

    The redis tier is here for the same reason and gets the same treatment:
    its own invariants — that an exact trim actually trims, that WRONGTYPE is a
    ResponseError, that XAUTOCLAIM's completion cursor is "0-0", that XINFO
    GROUPS' lag goes nil — are ones fakeredis answers WRONGLY rather than not at
    all, so a silently-skipped job leaves the suite asserting the fake's
    behaviour and nothing else.

    Shares tier_enabled() with test_pg_integration._PG_ENABLED and
    test_redis_integration._REDIS_ENABLED rather than re-deriving the check:
    the two used to be hand-kept in step, and a gate that disagrees with the
    skipif it gates is worse than no gate at all.

    Matches the marker as a WORD in the expression, not as the whole string.
    `-m pg` was the only spelling that reached this check, so the moment anyone
    narrowed a run — `-m "pg and not slow"`, `-m "pg or redis"` — the gate went
    silent and the all-skipped-green hole reopened under the exact command a
    developer reaches for when triaging.
    """
    enablers = {
        "pg": ("RUN_PG_TESTS", "POSTGRES_TEST_URL"),
        "redis": ("RUN_REDIS_TESTS", "REDIS_TEST_URL"),
    }
    markexpr = config.option.markexpr
    selected = set(re.findall(r"[A-Za-z_][A-Za-z0-9_]*", markexpr)) - {
        "and",
        "or",
        "not",
    }
    tiers = [t for t in enablers if t in selected]
    if len(tiers) != 1:
        # Zero tiers: an ordinary run, nothing to gate. More than one: the
        # expression selects a mix, so "every test would skip" is not what a
        # disabled tier means any more and the per-module skipif is the honest
        # reporter.
        return
    tier = tiers[0]
    flag, url = enablers[tier]
    if tier_enabled(flag, url):
        return
    print(
        f"\nERROR: `-m {markexpr}` selects the {tier} tier but it is disabled, "
        f"so every test would skip.\n       Set {flag}=1 (needs Docker) or "
        f"{url}.",
        file=sys.stderr,
    )
    raise pytest.UsageError(f"{tier} tier selected but not enabled")


@pytest.fixture(autouse=True)
def use_thread_ytdlp_pool(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run yt-dlp extraction on an in-process ThreadPoolExecutor.

    Production uses a ProcessPoolExecutor, which pickles the submitted callable to a
    worker. Tests patch src.youtube._ytdlp_extract with a MagicMock, which is unpicklable
    — and even a real patch would never reach a worker process. A thread pool runs
    in-process so the patch is honored and no children are ever spawned.

    A fresh YtdlpPool per test, not a shared one: a test that exercises the shutdown path
    (src.main.MusicBotApp.close) closes the pool permanently, and the next test must not
    inherit a closed one. ThreadPoolExecutor spawns threads lazily on first submit, so
    tests that never extract pay nothing.

    Consequence worth stating plainly: **no test that goes through src.youtube ever
    spawns a worker process.** The pickling that production performs on every submit is
    asserted directly and cheaply by TestProcessBoundaryContract (tests/test_youtube.py),
    and one dedicated test in tests/test_ytdlp_pool.py spawns a real worker end-to-end.
    Everything else about the process path — orphan reaping, live playback — is the
    manual gate in docs/YTDLP_POOL_ENCAPSULATION_PLAN.md §7.
    """
    from concurrent.futures import ThreadPoolExecutor

    import src.youtube as youtube
    from src.ytdlp_pool import YtdlpPool

    pool = YtdlpPool(
        max_workers=4,
        executor_factory=lambda: ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="ytdlp-test"
        ),
    )
    monkeypatch.setattr(youtube, "ytdlp_pool", pool)
    yield
    pool.shutdown(wait=False)


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

    loop() blocks on _restore_complete until _restore_state() finishes (see its docstring
    for why); since start() never runs here, nothing would set it. Tests that
    exercise that race explicitly should clear it again before calling loop().

    loop() then blocks on the playback gate until a voice connection is
    established (docs/PLAYBACK_GATE_PLAN.md). start() and the -join/-play call
    sites that open it never run here either, so it is opened for the same
    reason — tests that exercise the gate itself should clear it again.
    """
    mp = MusicPlayer(mock_bot, mock_guild, mock_channel, mock_ctx.cog, redis=fake_redis)
    mp._restore_complete.set()
    mp._playback_gate.set()
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
    from src.youtube import YTDL, YTDLVideoInfo

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
                # Arbitrary per-test overrides merge in above, so this is a plain
                # dict by construction; the cast is the info-dict shape assertion.
                data=cast(YTDLVideoInfo, default_data),
                requester=mock_author,
            )

    return _make


@pytest.fixture
def music_bot(mock_bot: MagicMock) -> MusicBot:
    """Minimal MusicBot instance bypassing __init__ Discord registration.

    Shared: tests/test_musicbot.py drives the cog's commands, tests/test_ping.py
    drives the -ping dashboard through the same cog.
    """
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog._spotify_status = SpotifyStatus.ENABLED
    cog.redis = None
    cog._active_spans = {}
    cog._alone_timers = {}
    cog._restore_tasks = set()
    return cog
