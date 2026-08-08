"""Shared fixtures for the discord-music-bot test suite."""

import re
import sys
import time
from typing import Any, Optional, cast
from collections.abc import AsyncIterator, Callable, Iterator
from unittest.mock import AsyncMock, MagicMock

import discord
import fakeredis
import pytest
import structlog
from fakeredis.model import StreamEntryKey, XStream
from redis.asyncio import Redis

from src.config import SpotifyStatus
from src.debug import RuntimeSampler
from src.musicbot import MusicBot
from src.musicplayer import MusicPlayer
from src.spotify import Spotify
from tests.helpers import noop_ffmpeg_init, tier_enabled


def pytest_collection_modifyitems(
    session: pytest.Session, config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Refuse a `-m pg` / `-m redis` run that would skip the entire tier.

    An all-skipped tier exits 0 and looks green while invariants that live only
    there stop being checked — ON CONFLICT dedup, the -history tie-break, the
    schema lock, and everything fakeredis answers WRONGLY rather than not at all.
    Matches the marker as a WORD so `-m "pg and not slow"` cannot slip past, and
    shares tier_enabled() with the modules' skipif so the two cannot disagree.
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

    Production pickles the submitted callable to a ProcessPoolExecutor worker; tests
    patch src.youtube._ytdlp_extract with MagicMocks that could never be pickled, so a
    thread pool is the only seam that honors the patch (fresh per test — the
    shutdown-path tests close it permanently). Consequence: no test through
    src.youtube spawns a worker process, so the pickle contract is asserted by
    TestProcessBoundaryContract and one test in tests/test_ytdlp_pool.py.
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


@pytest.fixture(autouse=True)
def scrub_config_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pin the env the suite asserts against, whatever shell it runs in.

    `just test` does not load .env, so the exposure is an EXPORTED variable: a
    shell carrying a real POSTGRES_URL silently disables every default-credential
    test, one exporting the default flips them the other way (setenv still wins).

    HISTORY_ARCHIVE_ENABLED is pinned TRUE, deliberately INVERTING the ship
    default: the enabled path exercises strictly more code (outbox XADD, notify,
    drainer wiring) and hundreds of assertions encode it. Don't change this to
    the ship default — disabled-mode tests monkeypatch the flag per case and win
    over this fixture (same MonkeyPatch instance, later call).

    DEBUG_MODE is scrubbed rather than pinned, because here the ship default is
    the one hundreds of embed assertions encode: with it on, every response grows
    a debug footer. Debug-on tests monkeypatch it (or set an override) per case.
    """
    monkeypatch.delenv("POSTGRES_URL", raising=False)
    monkeypatch.delenv("DEBUG_MODE", raising=False)
    monkeypatch.setenv("HISTORY_ARCHIVE_ENABLED", "true")


@pytest.fixture(autouse=True, scope="session")
def configure_structlog_for_tests() -> None:
    """Configure structlog with minimal output for tests.

    Plain renderer instead of JSON so output is readable; no OTel context
    processor (there is no TracerProvider in tests).
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

    Otherwise bind_contextvars(guild_id=...) leaks into later tests.
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
    # Set explicitly, not left to the spec. `spec=discord.Member` auto-mocks
    # guild_permissions, so every attribute on it answers with a TRUTHY MagicMock —
    # a permission check would pass here even if the code had deleted it. A test
    # that means "denied" has to say so by setting this False.
    member.guild_permissions = MagicMock(spec=discord.Permissions)
    member.guild_permissions.manage_guild = True
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
    # Owner by default: on a self-hosted bot the application owner is the
    # operator, so it is the ordinary case for owner-gated commands (today,
    # -ping's default-password advisory). AsyncMock is required — a bare
    # MagicMock attribute is unawaitable and every such command dies with TypeError.
    ctx.bot.is_owner = AsyncMock(return_value=True)
    # Explicit, for the same reason mock_author pins guild_permissions: a bare
    # MagicMock answers `.extras.get("anything")` with a truthy mock, so every
    # command would look like it carried every flag. cog_before_invoke reads
    # `extras["observation_only"]` to decide whether to skip get_mp(), and an
    # auto-mock there silently exempts the whole suite.
    ctx.command.extras = {}
    return ctx


@pytest.fixture
def mock_bot(mock_guild: MagicMock) -> MagicMock:
    bot = MagicMock()
    bot.guilds = [mock_guild]
    bot.latency = 0.05
    bot.is_closed.return_value = False
    bot.wait_until_ready = AsyncMock()
    # Declared, never auto-vivified: MusicPlayer.__init__ reads
    # bot.history_drainer, and an auto-vivified MagicMock answers `is not None`
    # with True — so every player would take the archive-enabled arm and the
    # disabled arm (the SHIP default) would be exercised by nothing. Dropping
    # musicplayer.py's None guard once passed the entire suite while bricking
    # every -play in the default configuration.
    bot.history_archive = MagicMock()
    bot.history_drainer = MagicMock()
    # No create_task mock needed — MusicPlayer.start() is never called in tests
    return bot


def _patch_xadd_monotonic_ids() -> None:
    """Make a generated stream ID outrank every ID the stream ever issued.

    fakeredis derives the next `*` ID from the LIVE entries alone, so a stream
    drained to empty forgets its last ID and reissues one that is not greater —
    same millisecond, or any wall-clock step backwards. XREADGROUP `>` then
    skips that entry forever, because it does not outrank the group's
    last-delivered-id, and the drainer reports an empty cycle over a non-empty
    outbox. Real Redis keeps last_id independently of the entries.

    Divergence 6 in tests/test_redis_integration.py; still present in fakeredis
    2.37.0. Pinned by TestGeneratedStreamIds in tests/test_history_archive.py.
    """
    original = XStream.add

    def add(self: XStream, fields: Any, entry_key: Any = "*", **kwargs: Any) -> Any:
        key = entry_key.decode() if isinstance(entry_key, bytes) else entry_key
        if key in (None, "*") and self._last_generated_id is not None:
            last = StreamEntryKey.parse_str(self._last_generated_id)
            if last.ts >= int(1000 * time.time()):
                entry_key = f"{last.ts}-{last.seq + 1}"
        return original(self, fields, cast(str, entry_key), **kwargs)

    XStream.add = add  # type: ignore[method-assign]


_patch_xadd_monotonic_ids()


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
    """MusicPlayer on fake Redis; start() is not called — tests drive state directly.

    loop() blocks on both _restore_complete and the playback gate, and nothing
    here would ever set them (start() and the -join/-play call sites never run),
    so both are set. Tests exercising either race must clear them again first.
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
                # Per-test overrides merge in above, so this is a plain dict by
                # construction; the cast is the info-dict shape assertion.
                data=cast(YTDLVideoInfo, default_data),
                requester=mock_author,
            )

    return _make


@pytest.fixture
def music_bot(mock_bot: MagicMock) -> MusicBot:
    """Minimal MusicBot instance bypassing __init__ Discord registration.

    Shared by tests/test_musicbot.py (commands) and tests/test_ping.py (dashboard).
    """
    cog = MusicBot.__new__(MusicBot)
    cog.bot = mock_bot
    cog.mps = {}
    cog.spotify = MagicMock()
    cog._spotify_status = SpotifyStatus.ENABLED
    cog.redis = None
    # None, not a mock: MusicBot declares __slots__, so an unset slot raises
    # AttributeError rather than returning None. Tests that care about the
    # Postgres row set their own archive (see TestPingReportsPostgres).
    cog.history_archive = None
    cog._active_spans = {}
    cog._alone_timers = {}
    cog._restore_tasks = set()
    cog._enqueue_locks = {}
    # Off, matching the ship default and the DEBUG_MODE scrub above: with debug
    # mode on, every response grows a footer and the suite's embed assertions
    # would be asserting against decorated output everywhere.
    cog._debug_default = False
    cog._debug_overrides = {}
    # Same shape __init__ builds: the toggle stamps these so a hydration pass that
    # read before it cannot apply an older value on top.
    cog._debug_toggle_seq = 0
    cog._debug_toggled_at = {}
    cog._debug_unpersisted = set()
    cog._runtime_sampler = RuntimeSampler()
    return cog
