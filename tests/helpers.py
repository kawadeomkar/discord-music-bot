"""Shared test helper functions.

Plain module-level functions (not fixtures) so any test file or conftest can
import them directly, without routing through pytest's plugin machinery.
"""

import asyncio
import contextlib
import dataclasses
from contextlib import AbstractContextManager
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Optional, TypedDict, cast
from collections.abc import AsyncGenerator, Callable, Coroutine, Iterator
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import discord
from discord.ext import commands
from discord.utils import MISSING as _DISCORD_MISSING

from src.guild_queue import GuildQueue
from src.guild_state import GuildConfig
from src.queue_item import QueueObject
from src.redis_client import GuildRedisStore, iter_guild_configs
from src.settings import GuildSettings
from src.play_placement import PlayMode, PlayRequest
from src.youtube import YTDL, YoutubePlaylist

if TYPE_CHECKING:
    from src.musicbot import MusicBot


def admit(
    music_bot: MusicBot,
    ctx: MagicMock,
    mp: Any,
    *,
    mode: PlayMode = PlayMode.NORMAL,
    query: str = "test",
) -> PlayRequest:
    """Register a request as -play would, so a helper that inserts under the
    place lock can be called directly. A bare MagicMock answers `retired` with a
    truthy mock, which place() reads as torn down; pin it as mock_mp() does."""
    if isinstance(mp, MagicMock) and isinstance(mp.retired, MagicMock):
        mp.retired = False
    return music_bot._plays.register(ctx, query=query, mp=mp, mode=mode)


async def settle(ticks: int = 12) -> None:
    """Let every runnable task reach its next suspension point."""
    for _ in range(ticks):
        await asyncio.sleep(0)


def song(n: int, ctx: MagicMock) -> QueueObject:
    return QueueObject(
        webpage_url=f"https://yt.com/v={n}", title=f"Song {n}", requester=ctx.author
    )


@contextlib.contextmanager
def recording_span() -> Iterator[MagicMock]:
    """A span double in place of the current span. Its context is invalid so
    structlog's OTel processor, which reads the same global, skips the trace-id
    format it would otherwise apply to a MagicMock."""
    with patch("src.musicbot.trace.get_current_span") as current:
        span = current.return_value
        span.get_span_context.return_value.is_valid = False
        yield span


def command_callback(
    command: commands.Command[Any, ..., Any],
) -> Callable[..., Coroutine[Any, Any, Any]]:
    """Return a command's raw callback, invocable as ``callback(cog, ctx, ...)``.

    ``Command.callback`` is a union of the cog-bound and unbound signatures and a
    call must satisfy *both*, so passing the cog explicitly (correct at runtime)
    can never type-check. The cast collapses it to the shape callers use."""
    return cast(Callable[..., Coroutine[Any, Any, Any]], command.callback)


def mocked(obj: object) -> MagicMock:
    """The MagicMock behind an attribute that production types as the real thing.

    Fixtures hand MusicPlayer/MusicBot mocks declared as `discord.Guild` /
    `commands.Bot`, so reading `.side_effect` or assigning a read-only property is
    right at runtime and rejected statically. The parameter is `object`, not
    `MagicMock`, precisely so it accepts the production-typed expression."""
    return cast(MagicMock, obj)


def described(embed: discord.Embed) -> str:
    """An embed's description, asserted non-empty.

    `.description` is `Optional[str]` and every assertion means "present, and it
    says X"; failing the first half separately says which of the two broke."""
    assert embed.description is not None
    return embed.description


def queue_object(item: object) -> QueueObject:
    """A queue entry narrowed to `QueueObject`.

    `display_items()` is typed `list[QueueObject]`, so this only narrows the
    `object` a test holds; it is the assertion that the narrowing is true."""
    assert isinstance(item, QueueObject)
    return item


def noop_ffmpeg_init(self: Any, *args: Any, **kwargs: Any) -> None:
    """Replacement for FFmpegOpusAudio.__init__ that stubs all pre-spawn attributes.

    Patching out the real __init__ leaves the sentinels unassigned, so GC's
    __del__ → cleanup() → _kill_process() reads them and raises AttributeError.
    These mirror discord.py's pre-spawn state, so every guard returns early."""
    self._process = _DISCORD_MISSING
    self._stopped = False
    self._stdout = None
    self._stdin = None
    self._stderr = None


def stub_create_task(return_value: Optional[Any] = None) -> MagicMock:
    """Return a mock that replaces loop.create_task or asyncio.create_task.

    A plain MagicMock(return_value=...) parks the coroutine in call_args unclosed,
    raising "coroutine was never awaited" on GC. This closes each one immediately
    and returns a configurable mock Task so return-value assertions pass."""

    def _impl(coro: Coroutine[Any, Any, Any]) -> Any:
        coro.close()
        return return_value if return_value is not None else MagicMock()

    return MagicMock(side_effect=_impl)


def stub_yt_playlist(
    tracks: list[QueueObject], *, title: Optional[str] = None, unavailable: int = 0
) -> AsyncMock:
    """A stand-in for YTDL.yt_playlist resolving to `tracks`, untitled and with
    nothing unavailable unless the test says otherwise."""
    return AsyncMock(
        return_value=YoutubePlaylist(
            title=title, tracks=tracks, unavailable=unavailable
        )
    )


def make_mock_task() -> MagicMock:
    """A MagicMock resembling a running asyncio.Task, for cancellation asserts."""
    task = MagicMock(spec=asyncio.Task)
    task.done.return_value = False
    task.cancel = MagicMock()
    return task


def tier_enabled(*env_vars: str) -> bool:
    """Is an opt-in integration tier turned on by its environment?

    One definition for all three readers (conftest's gate hook, both tiers' own
    skipif) — a gate disagreeing with what it gates is worse than no gate. "0",
    "false" and "" are disabled; `bool(os.getenv(...))` reads all three as on."""
    import os

    return any(
        os.getenv(name, "").strip().lower() not in ("", "0", "false", "no")
        for name in env_vars
    )


def bind_loopback_only(container: Any, port: int) -> None:
    """Publish `port` on 127.0.0.1 instead of every interface.

    Docker binds 0.0.0.0, so a plain testcontainer exposes an unauthenticated
    Redis / weak-password Postgres to the LAN for a whole run. testcontainers
    hands `.ports` to docker-py, which accepts `(host_ip, host_port)`; `None`
    keeps the random high port. Call before start — that dict is read once."""
    container.ports[port] = ("127.0.0.1", None)


def passthrough_prefetch() -> AsyncMock:
    """A `YTDL.prefetch_stream` double honouring its contract: the item it was
    handed comes back as warmed. A bare AsyncMock answers a Mock, which the
    callers would swap into the queue in the item's place."""
    return AsyncMock(side_effect=lambda qo, *, redis=None: qo)


def seed_queue(gq: GuildQueue, *items: QueueObject) -> None:
    """Queue items without touching Redis — `put()` minus the mirror.

    Synchronous, so the sync tests (embeds, ETA) can use it too. Nothing here
    claims: a test wanting an in-flight head calls `get()`, as production does.
    """
    gq._items.extend(items)
    gq._sync_wake()


def no_typing(target: str) -> AbstractContextManager[MagicMock]:
    """Stub a module's background_typing with an inert async CM.

    `target` is the MODULE the command under test resolves the name in, not where
    it is defined, and it has no default: every command body owns its own reference
    now, so a wrong module leaves the real keepalive running instead of failing.

    TestPlayCommand needs this because it patches asyncio.create_task as a join-task
    spy, and the typing keepalive would otherwise hit the same patch, polluting call
    counts and taking the fake join future. The wrapper itself is covered by
    TestBackgroundTyping."""
    return patch(target, MagicMock(return_value=contextlib.nullcontext()))


def no_slow_notice(target: str) -> AbstractContextManager[MagicMock]:
    """Stub a module's slow_resolve_notice with an inert async CM.

    The same trap no_typing names, for the same reason: the notice arms a delayed
    poster with asyncio.create_task, which TestPlayCommand's join-task spy would
    otherwise catch. Paired with no_typing wherever that spy runs. The notice
    itself is covered by TestSlowResolveNotice."""
    return patch(target, MagicMock(return_value=contextlib.nullcontext()))


def in_authors_channel(vc: MagicMock, ctx: Optional[MagicMock]) -> MagicMock:
    """Seat a voice-client double in the author's channel, or somewhere else. Queue
    control is gated on the bot being in the author's channel at dispatch AND at
    the insert, and a double with no channel reads as "somewhere else"."""
    vc.channel = (
        ctx.author.voice.channel
        if ctx is not None
        else MagicMock(spec=discord.VoiceChannel)
    )
    return vc


def connected_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client, nothing playing — what a successful cold join leaves
    behind. is_connected is explicit: the cold path checks it, because discord.py
    registers the client on the guild before the handshake completes."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return in_authors_channel(vc, ctx)


def playing_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client, actively playing. Both flags must be set explicitly:
    an unstubbed is_paused() returns a truthy Mock, silently sending -play down the
    interjection branch instead of the append path."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = True
    vc.is_paused.return_value = False
    vc.is_connected.return_value = True
    return in_authors_channel(vc, ctx)


def paused_vc(ctx: Optional[MagicMock] = None) -> MagicMock:
    """Connected voice client with a song parked paused. is_connected is explicit:
    -resume's rejoin checks it, and an auto-vivified one answers True by accident
    rather than by choice."""
    vc = MagicMock(spec=discord.VoiceClient)
    vc.is_playing.return_value = False
    vc.is_paused.return_value = True
    vc.is_connected.return_value = True
    return in_authors_channel(vc, ctx)


MOCK_QUEUED_ROWS = "`1` [**Row**](https://x) · `3:00` · Est. playing at **9:41 PM PDT**"


def mock_mp(qsize: int = 0) -> MagicMock:
    """MusicPlayer stand-in for the -play cold path, with the playback-gate
    hooks awaitable: play() takes defer_playback() as an async context manager
    and awaits wait_for_restore() before front-inserting."""
    mp = MagicMock()
    mp.defer_playback = MagicMock(return_value=contextlib.nullcontext())
    mp.wait_for_restore = AsyncMock(return_value=True)
    # Numeric, not auto-vivified: _abandon_cold_start COMPARES this, and a Mock
    # raises TypeError there rather than answering.
    mp.playback_holds = 1  # the hold this command itself takes
    # Explicit, not auto-vivified: place() reads it, and a MagicMock is truthy.
    mp.retired = False
    mp.queue.generation = 0
    mp.repark_crashed_head = AsyncMock()
    # Awaitable, not auto-vivified: interject_flow settles the prefetch BEFORE it
    # takes the place lock, and a bare Mock is not awaitable there.
    mp.settle_prefetch = AsyncMock()
    mp.queue_put_front = AsyncMock()
    mp.queue_put = AsyncMock()
    # `--next` inserts through its own wrapper, which neutralizes the loop's
    # prefetch first — a plain front insert lands behind that claim.
    mp.queue_put_next = AsyncMock()
    # A str, not auto-vivified: the queued-playlist card joins it into its text.
    mp.playlist_facts = MagicMock(return_value="")
    # The card's rows, for the same reason — and recognizable, so a command-level
    # test can assert the card carries them rather than a Mock's repr.
    mp.queued_rows = MagicMock(return_value=MOCK_QUEUED_ROWS)
    # An int, not auto-vivified: the card derives "Songs ahead" from it. Mirrors
    # the real lookup's fallback, which is the depth the insert saw.
    mp.queued_slot = MagicMock(side_effect=lambda _tracks, *, ahead: ahead + 1)
    mp.queue.claim_outstanding = MagicMock(return_value=False)
    mp.queue.qsize = MagicMock(return_value=qsize)
    # Numeric for the same reason as playback_holds: this lands in
    # QueueObject.queue_position and rides to Postgres through HistoryEntry's
    # integer clamp, which a Mock raises on rather than answering.
    mp.enqueue_depth = MagicMock(return_value=qsize)
    # Numeric for that reason too: _cold_start_left_something_playable compares it
    # to decide whether a late refusal may disconnect the session, and an
    # auto-vivified Mock is truthy — which spares every teardown these tests pin.
    mp.queue.display_size = MagicMock(return_value=qsize)
    # Mirrors the real builder's contract: a notice only when the restore
    # actually left something in the queue (see build_resume_notice_embed).
    mp.build_resume_notice_embed = MagicMock(
        return_value=discord.Embed(title="❗ Resumed from queue") if qsize else None
    )
    return mp


# The ask fields, taken from the source itself: a real YTDL answers each of these
# off the queue object it holds, so a double must too. Reflective, so a field
# added to QueueObject and exposed on YTDL is wired below without an edit here.
ASK_FIELDS: tuple[str, ...] = tuple(
    f.name
    for f in dataclasses.fields(QueueObject)
    if isinstance(getattr(YTDL, f.name, None), property)
)


def same_ask(a: QueueObject, b: QueueObject) -> bool:
    """Whether two items are one ask field for field, `queue_position` aside: a
    put re-mints the depth, so the item it takes is a copy of the one the caller
    built."""
    return all(
        getattr(a, f.name) == getattr(b, f.name)
        for f in dataclasses.fields(QueueObject)
        if f.name != "queue_position"
    )


def unresolved(term: str, requester: Any = None, **fields: Any) -> QueueObject:
    """A queue item still waiting to resolve, the way a Spotify collection track is
    queued by a walk that named no display row: `search` holds the term and the
    display fields arrive with the resolve. `title` and `webpage_url` are the two
    a walk DOES supply rows for, so both are keywords here and both default empty."""
    return QueueObject(
        webpage_url=fields.pop("webpage_url", ""),
        title=fields.pop("title", ""),
        requester=requester if requester is not None else stub_requester(),
        search=f"ytsearch:{term}",
        **fields,
    )


def stub_requester(user_id: int = 4242, name: str = "Loop User") -> MagicMock:
    """A user stand-in carrying the real values a play's ask serializes: the
    queue mirror HSETs the id and play_history stores the display name, and a
    MagicMock attribute is neither HSET-able nor clampable."""
    who = MagicMock()
    who.id = user_id
    who.display_name = name
    who.mention = f"<@{user_id}>"
    return who


def give_queue_object(song: Any, queued: QueueObject) -> QueueObject:
    """Give a YTDL double the queue object a real source holds, with the ask
    fields wired through it as YTDL's properties are.

    A test that assigns one (`song.played_at = 1.0`) moves it on the entry, so a
    rebuild reading `song.queued` sees what the test set — the divergence that
    otherwise hides a rebuild dropping a field. PropertyMock goes on the type
    because mock stores attributes on the instance; every Mock has a type of its
    own, so this reaches no other double.

    The write is a `replace()`, so `song.queued` is a new object afterwards where
    the real setter mutates the entry it holds; that contract is pinned on a real
    source by test_youtube.py's test_the_start_stamp_reaches_the_entry."""
    song.queued = queued
    for name in ASK_FIELDS:

        def access(*value: Any, _name: str = name) -> Any:
            if value:
                song.queued = replace(song.queued, **{_name: value[0]})
                return None
            return getattr(song.queued, _name)

        setattr(type(song), name, PropertyMock(side_effect=access))
    return queued


class Ask(TypedDict):
    """The two ask-time analytics keywords a queue item, the resolve functions
    and -replay take together, spelled once per test module and splatted."""

    queued_at: float
    queue_position: int


def ask_of(item: Any) -> Ask:
    """The ask-time analytics an item or a song double carries, as the keywords
    it was built from."""
    return {"queued_at": item.queued_at, "queue_position": item.queue_position}


# What -replay mints at dispatch: the command message's snowflake time, and
# depth 0 — the replay plays immediately.
REPLAY_ASK: Ask = {"queued_at": 1752530500.5, "queue_position": 0}


def loop_song(url: str, title: str, *, position: float) -> MagicMock:
    """A spec'd YTDL stand-in — a bare MagicMock reads truthy for
    start_paused/is_resume and would trip the loop's start path."""
    song = MagicMock(spec=YTDL)
    song.title = title
    song.webpage_url = url
    song.duration_secs = 210
    song.duration_label = "0:03:30"
    song.uploader = "Loop Channel"
    song.thumbnail = ""
    song.views = None
    song.likes = None
    song.abr = None
    song.asr = None
    song.acodec = ""
    song.start_offset = 0
    song.position_secs = position
    song.produced_audio = True
    song.data = {}
    give_queue_object(
        song, QueueObject(webpage_url=url, title=title, requester=stub_requester())
    )
    return song


def replayed_song(source: QueueObject) -> MagicMock:
    """The source a -replay's copy resolves to, HOLDING the entry that was
    queued rather than a copy of a few of its fields — so a test can put state
    on `source` and see what the play reads back off it, including the fields
    _neutralize_prefetch's rebuild carries."""
    song = loop_song(source.webpage_url, source.title, position=42.0)
    give_queue_object(song, source)
    return song


async def stored_config(store: GuildRedisStore) -> GuildConfig:
    """The store's config as read_config returns it, for a read that must land."""
    config = await store.read_config()
    assert config is not None
    return config


async def read_all_configs(
    redis: Any, guild_ids: list[int], **kwargs: Any
) -> dict[int, GuildConfig]:
    """Every batch iter_guild_configs yields for `guild_ids`, merged."""
    return {
        guild_id: config
        async for batch in iter_guild_configs(redis, guild_ids, **kwargs)
        for guild_id, config in batch.items()
    }


@contextlib.contextmanager
def stalled_config_reads() -> Iterator[None]:
    """A Redis that accepts a guild config read and never answers it: the single
    read and the batched one both park for good."""
    never = asyncio.Event()

    async def _stall(*_args: Any, **_kwargs: Any) -> None:
        await never.wait()

    async def _stall_batches(*_args: Any, **_kwargs: Any) -> AsyncGenerator[Any]:
        await never.wait()
        yield {}

    with (
        patch.object(GuildRedisStore, "read_config", new=_stall),
        patch("src.settings.iter_guild_configs", new=_stall_batches),
    ):
        yield


def add_settings_state(cog: Any) -> None:
    """The settings state MusicBot.__init__ builds, for a cog built without it."""
    cog.guild_settings = GuildSettings(cog)
    cog._hydrate_retries = set()
    cog._orphan_sweep_claimed = False


def members(cls: type) -> set[str]:
    """The values of a constants class's UPPER_CASE attributes."""
    return {v for k, v in vars(cls).items() if k.isupper()}
