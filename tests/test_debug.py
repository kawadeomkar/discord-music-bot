"""Tests for src/debug.py — argument parsing, the config redaction table, and the
snapshot's collectors. The redaction tests are the load-bearing ones: this embed is
built to be screenshotted into an issue, so a value that should never render is a
credential leak rather than a cosmetic bug."""

import asyncio
import os
import re
import time
from pathlib import Path
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import orjson
import pytest
from redis.asyncio import Redis
from discord.ext import commands
from opentelemetry import trace as trace_api

from src import config, debug
from src.guild_state import GuildConfig
from src.history_archive import ArchiveStats
from src.redis_client import GuildRedisStore
from src.musicbot import MusicBot as MusicBotCog
from src.util import spawn_background
from tests.helpers import command_callback
from src.musicplayer import MusicPlayer
from src.debug import (
    DebugAction,
    DebugInputs,
    _CONFIG_ALLOWLIST,
    _ConfigKind,
    _codeblock_fields,
    run_debug_dashboard,
    config_lines,
    discord_lines,
    guild_lines,
    parse_debug_arg,
    redact_url,
    render_config_value,
    unknown_arg_message,
)


async def _snapshot(ctx: MagicMock, inputs: DebugInputs) -> discord.Embed:
    """Drive the live dashboard to completion and return the FINAL embed — what a
    user is looking at once every block has landed. The dashboard sends a skeleton
    and then edits, so the last edit (or the send, when nothing was deferred) is the
    finished card."""
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    ctx.channel.send = AsyncMock(return_value=message)
    await run_debug_dashboard(ctx, inputs)
    call = message.edit.await_args or ctx.channel.send.await_args
    assert call is not None
    return cast(discord.Embed, call.kwargs["embeds"][0])


async def _skeleton(ctx: MagicMock, inputs: DebugInputs) -> discord.Embed:
    """The FIRST embed the user sees, before any deferred block has landed."""
    message = MagicMock(spec=discord.Message)
    message.edit = AsyncMock()
    ctx.channel.send = AsyncMock(return_value=message)
    await run_debug_dashboard(ctx, inputs)
    call = ctx.channel.send.await_args
    assert call is not None
    return cast(discord.Embed, call.kwargs["embeds"][0])


def _archive_stats() -> ArchiveStats:
    return ArchiveStats(
        database_bytes=1,
        table_bytes=1,
        rows_estimate=1,
        rejected_estimate=0,
        connections=1,
        max_connections=100,
        shared_buffers="128MB",
        cache_hit_ratio=0.99,
    )


def _pg_stats(**overrides: Any) -> ArchiveStats:
    """A realistic full sample for the renderer tests; override per case. The
    counter values are the `before` baseline — pair with a second call carrying
    deltas and a later `monotonic` to exercise the windowed rates."""
    base: dict[str, Any] = dict(
        database_bytes=134217728,
        table_bytes=100663296,
        rows_estimate=412083,
        rejected_estimate=0,
        connections=3,
        max_connections=100,
        shared_buffers="128MB",
        cache_hit_ratio=0.994,
        active_backends=0,
        active_io_wait=0,
        active_lock_wait=0,
        active_other_wait=0,
        active_time_ms=5_000.0,
        xacts_total=1_000,
        tuples_total=50_000,
        blks_hit=9_000,
        blks_read=1_000,
        temp_bytes=0,
        deadlocks=0,
        monotonic=100.0,
    )
    base.update(overrides)
    return ArchiveStats(**base)


class TestParseDebugArg:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("", DebugAction.STATUS),
            ("   ", DebugAction.STATUS),
            ("--enable", DebugAction.ENABLE),
            ("  --ENABLE ", DebugAction.ENABLE),
            ("--disable", DebugAction.DISABLE),
        ],
    )
    def test_accepted_forms(self, raw: str, expected: DebugAction) -> None:
        assert parse_debug_arg(raw) is expected

    @pytest.mark.parametrize(
        "raw", ["enable", "-enable", "--enabled", "--enable please", "on", "--verbose"]
    )
    def test_everything_else_is_unparseable(self, raw: str) -> None:
        assert parse_debug_arg(raw) is None

    @pytest.mark.parametrize("raw", ["enable", "DISABLE", " disable "])
    def test_missing_dashes_gets_a_pointed_hint(self, raw: str) -> None:
        """The realistic typo. A FlagConverter would have answered this with a
        FlagError raised before the body ever ran, which is why the parse is by hand."""
        assert f"--{raw.strip().lower()}" in unknown_arg_message(raw)

    def test_unknown_argument_is_never_echoed_back(self) -> None:
        """User text rendered into an embed the BOT sends is a mention-injection
        surface — discord.py's default allowed_mentions permits @everyone."""
        message = unknown_arg_message("@everyone `oops`")
        assert "@everyone" not in message
        assert debug.DEBUG_USAGE in message


class TestRedactUrl:
    def test_userinfo_is_replaced(self) -> None:
        assert (
            redact_url("postgresql://bot:hunter2@db.internal:5432/musicbot")
            == "postgresql://***@db.internal:5432/musicbot"
        )

    def test_credentialless_url_is_untouched(self) -> None:
        assert redact_url("redis://localhost:6379") == "redis://localhost:6379"

    @pytest.mark.parametrize(
        "raw",
        [
            "collector.prod.internal.corp:4317",
            "pot-sidecar.prod.internal.corp:4416",
            "cache.internal:6379",
            # One slash, so there is no authority to rewrite AND the host lands in
            # .path — replacing .netloc alone leaves it visible.
            "http:/host.internal:9090/x",
        ],
    )
    def test_hide_host_does_not_fail_open_on_a_schemeless_url(self, raw: str) -> None:
        """urlsplit only fills .netloc when it sees `//`. Without it the host lands
        in .scheme/.path, so rewriting .netloc wrote `***` into a slot that was never
        carrying the host — publishing internal topology beside a marker that reads
        as "this was redacted". Operators DO write these scheme-less; ping.probe_otel
        carries the same normalisation and says so."""
        redacted = redact_url(raw, hide_host=True)
        assert "internal" not in redacted
        assert "4317" not in redacted and "4416" not in redacted
        assert "6379" not in redacted and "9090" not in redacted

    def test_hide_host_still_keeps_the_shape_of_a_normal_url(self) -> None:
        """The guard above must not flatten the ordinary case: an operator still
        needs to see which protocol is configured."""
        assert (
            redact_url("redis://user:pw@cache.internal:6379/0", hide_host=True)
            == "redis://***/0"
        )

    @pytest.mark.parametrize(
        "key", ["password", "api_key", "auth", "X-Token", "client_secret"]
    )
    def test_credential_query_values_are_replaced(self, key: str) -> None:
        """asyncpg honours `?password=` as readily as userinfo, so the query string
        is the second place a DSN hides a credential."""
        redacted = redact_url(f"postgresql://db/musicbot?{key}=hunter2&sslmode=require")
        assert "hunter2" not in redacted
        assert "***" in redacted
        assert "sslmode=require" in redacted

    def test_a_fragment_is_marked_rather_than_echoed(self) -> None:
        """urlsplit puts everything after a stray `#` into .fragment, so it is the
        one component neither the userinfo nor the query redaction reaches. Marked
        rather than dropped, so the row still says something was there."""
        redacted = redact_url("redis://host:6379/0#pw=hunter2")
        assert redacted == "redis://host:6379/0#***"
        assert "hunter2" not in redacted

    def test_unparseable_url_degrades(self) -> None:
        assert redact_url("http://[::1") == "unparseable"

    @pytest.mark.parametrize(
        "url",
        [
            "redis://u:pw@host:99999/0",  # out of range
            "redis://u:pw@host:abc/0",  # not a number
        ],
    )
    def test_a_bad_port_degrades_instead_of_raising(self, url: str) -> None:
        """`.port` is LAZY: urlsplit accepts these and raises on dereference. With
        the guard around the parse alone, the ValueError escaped past it — and since
        the whole Config block is one collector, one typo'd port in .env replaced all
        of it with "unavailable (ValueError)" exactly while someone was diagnosing
        that host.
        """
        assert redact_url(url) == "unparseable"
        assert "pw" not in redact_url(url)


class TestConfigAllowlist:
    def test_credentials_render_presence_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        for var in _CONFIG_ALLOWLIST:
            if var.kind is not _ConfigKind.SECRET:
                continue
            monkeypatch.setenv(var.name, "super-secret-value")
            rendered = render_config_value(var)
            assert rendered == "set"
            monkeypatch.delenv(var.name)
            assert render_config_value(var) == "unset"

    def test_every_credential_bearing_var_is_declared_secret(self) -> None:
        """The allowlist is decided by hand rather than by pattern-match, so this
        pins the review decision: these four carry credentials and must never
        render a value, POSTGRES_URL because it embeds the password in its DSN."""
        secrets = {v.name for v in _CONFIG_ALLOWLIST if v.kind is _ConfigKind.SECRET}
        assert secrets == {
            "DISCORD_TOKEN",
            "SPOTIFY_CLIENT_ID",
            "SPOTIFY_CLIENT_SECRET",
            "POSTGRES_URL",
        }

    def test_url_values_are_redacted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Credential AND host. The card is posted in the channel the operator typed
        in, so `cache:6379` is internal topology published to everyone there."""
        monkeypatch.setenv("REDIS_URL", "redis://user:pw@cache:6379/0")
        var = next(v for v in _CONFIG_ALLOWLIST if v.name == "REDIS_URL")
        rendered = render_config_value(var)
        assert "pw" not in rendered
        assert "cache" not in rendered and "6379" not in rendered
        assert rendered == "redis://***/0"

    def test_a_local_host_is_redacted_too(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unconditionally, or the presence of `***` would itself say "remote"."""
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        var = next(v for v in _CONFIG_ALLOWLIST if v.name == "REDIS_URL")
        assert "localhost" not in render_config_value(var)

    def test_unset_shows_the_value_in_force(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unset knob is not "nothing" — it is the default, and the default is
        what an operator is validating against."""
        monkeypatch.delenv("YTDLP_POOL_WORKERS", raising=False)
        var = next(v for v in _CONFIG_ALLOWLIST if v.name == "YTDLP_POOL_WORKERS")
        assert render_config_value(var) == "4 (default)"

    def test_unlisted_variable_never_renders(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The allowlist's whole point: a knob added later opts in at review time,
        so a future secret cannot leak by simply existing."""
        monkeypatch.setenv("SOME_FUTURE_SECRET", "leaked")
        assert "SOME_FUTURE_SECRET" not in "\n".join(config_lines())
        assert "leaked" not in "\n".join(config_lines())

    def test_no_duplicate_rows(self) -> None:
        names = [v.name for v in _CONFIG_ALLOWLIST]
        assert len(names) == len(set(names))


class TestCodeblockFields:
    def test_short_block_is_one_field(self) -> None:
        fields = _codeblock_fields("Config", ["a", "b"])
        assert fields == [("Config", "```\na\nb\n```")]

    def test_long_block_splits_rather_than_truncating(self) -> None:
        """Discord's field cap is 1024. A config listing clipped in place would
        read as a complete one, which is worse than showing none."""
        lines = [f"KNOB_{i:03d}  value" for i in range(120)]
        fields = _codeblock_fields("Config", lines)
        assert len(fields) > 1
        assert all(len(value) <= 1024 for _, value in fields)
        assert fields[1][0] == "Config (cont.)"
        rendered = "".join(value for _, value in fields)
        for line in lines:
            assert line in rendered


class TestDiscordBlock:
    def test_falls_back_to_single_shard_shape(self, mock_bot: MagicMock) -> None:
        """`latencies` is AutoShardedClient-only; a plain Bot (and every MagicMock,
        which auto-vivifies it into something that is not a list) takes this arm."""
        lines = "\n".join(discord_lines(mock_bot, players=2))
        assert "shards       1" in lines
        assert "#0 50 ms" in lines
        assert "players      2" in lines

    def test_renders_every_shard(self, mock_bot: MagicMock) -> None:
        mock_bot.latencies = [(0, 0.041), (1, 0.038)]
        lines = "\n".join(discord_lines(mock_bot, players=0))
        assert "shards       2" in lines
        assert "#0 41 ms, #1 38 ms" in lines

    def test_reconnecting_shard_is_not_rendered_as_a_number(
        self, mock_bot: MagicMock
    ) -> None:
        mock_bot.latencies = [(0, float("nan"))]
        assert "reconnecting" in "\n".join(discord_lines(mock_bot, players=0))


class TestGuildBlock:
    def _inputs(self, player: object = None) -> DebugInputs:
        return DebugInputs(
            debug_enabled=True,
            debug_overridden=True,
            players=1,
            player=cast(Optional[MusicPlayer], player),
        )

    def test_reports_no_player_and_no_voice(self, mock_guild: MagicMock) -> None:
        mock_guild.voice_client = None
        lines = "\n".join(
            guild_lines(mock_guild, self._inputs(), source="saved here"),
        )
        assert "player       no" in lines
        assert "voice        not connected" in lines
        assert "debug        on (saved here)" in lines

    def test_reports_player_queue_and_volume(
        self, mock_guild: MagicMock, music_player: MagicMock
    ) -> None:
        mock_guild.voice_client = None
        music_player.volume = 0.5
        lines = "\n".join(
            guild_lines(
                mock_guild, self._inputs(player=music_player), source="saved here"
            )
        )
        assert "player       yes" in lines
        assert "queue        0 queued" in lines
        assert "volume       50%" in lines

    @pytest.mark.parametrize(
        "name",
        [
            "```[Verify your account](https://evil.example)```",
            "`` ` ``",
            "```\n@everyone\n```",
            "x" * 200,
        ],
    )
    def test_a_channel_name_cannot_break_out_of_the_code_fence(
        self, mock_guild: MagicMock, name: str
    ) -> None:
        """A voice-channel name is the only user-controlled string in the snapshot,
        and `This server` is the block every guild member can see. A backtick run
        closes the fence, and Discord renders masked links inside embed field values
        — a clickable attacker link inside a card carrying the bot's name and avatar.
        Join-to-create bots hand ordinary members naming rights, so it needs no
        permission at all."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock(spec=discord.VoiceChannel)
        vc.channel.name = name
        vc.average_latency = 0.02
        mock_guild.voice_client = vc

        rendered = "\n".join(guild_lines(mock_guild, self._inputs(), source="x"))

        assert "`" not in rendered
        # The cap is half the control: an unbounded name pushes the rest of the
        # block past Discord's 1024-char field limit and the send 400s.
        assert max(len(line) for line in rendered.splitlines()) < 120

    def test_voice_latency_warming_up_is_not_an_error(
        self, mock_guild: MagicMock
    ) -> None:
        """discord.py #6430: both reads are inf, and average_latency raises
        ZeroDivisionError, until the first ACK ~5s after joining — exactly the
        window a tester looks at, since testing starts with a fresh join."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock(spec=discord.VoiceChannel)
        vc.channel.name = "General"
        vc.is_playing.return_value = False
        vc.is_paused.return_value = False
        vc.latency = float("inf")
        type(vc).average_latency = property(
            lambda _: (_ for _ in ()).throw(ZeroDivisionError())
        )
        mock_guild.voice_client = vc
        lines = "\n".join(guild_lines(mock_guild, self._inputs(), source="saved here"))
        assert "voice ws     warming up" in lines

    def test_missing_voice_permissions_are_flagged(self, mock_guild: MagicMock) -> None:
        vc = MagicMock(spec=discord.VoiceClient)
        vc.channel = MagicMock(spec=discord.VoiceChannel)
        vc.channel.name = "General"
        vc.is_playing.return_value = True
        vc.is_paused.return_value = False
        vc.latency = 0.021
        vc.average_latency = 0.023
        perms = MagicMock(spec=discord.Permissions)
        perms.connect = True
        perms.speak = False
        vc.channel.permissions_for.return_value = perms
        mock_guild.voice_client = vc
        lines = "\n".join(guild_lines(mock_guild, self._inputs(), source="saved here"))
        assert "voice        General · playing" in lines
        assert "voice ws     21 ms (avg 23 ms)" in lines
        assert "connect ✅" in lines
        assert "speak ⚠️" in lines


class TestSafeBlock:
    async def test_a_broken_collector_does_not_take_the_embed_down(
        self, mock_ctx: MagicMock
    ) -> None:
        """The degrade principle: every collector is wrapped, because a debug tool
        that crashes is worse than no debug tool."""
        mock_ctx.guild = None
        with patch.object(debug, "build_lines", side_effect=RuntimeError("boom")):
            embed = await _snapshot(
                mock_ctx,
                DebugInputs(
                    debug_enabled=False,
                    debug_overridden=False,
                    players=0,
                    operator=True,
                ),
            )
        build = next(f for f in embed.fields if f.name == "Build")
        assert "unavailable (RuntimeError)" in (build.value or "")
        assert [f.name for f in embed.fields].count("Config") == 1


class TestTheSnapshotDoesNotWaitForItsIO:
    """-debug went from collect-everything-then-send to skeleton-then-fill. The
    point is that a sick dependency delays one BLOCK, not the whole reply."""

    @staticmethod
    def _operator(**kw: Any) -> DebugInputs:
        return DebugInputs(
            debug_enabled=False, debug_overridden=False, players=0, operator=True, **kw
        )

    async def test_the_skeleton_shows_placeholders_for_blocks_still_collecting(
        self, mock_ctx: MagicMock
    ) -> None:
        """The runtime and redis probes each own a half-second window, so none of
        their blocks can have landed by the first send."""
        mock_ctx.guild.voice_client = None
        embed = await _skeleton(mock_ctx, self._operator())
        rendered = {f.name: (f.value or "") for f in embed.fields}
        for probe in ("runtime", "redis"):
            for name in debug._PROBE_BLOCKS[probe]:
                assert "collecting" in rendered[name], name

    async def test_a_block_that_can_answer_instantly_skips_the_placeholder(
        self, mock_ctx: MagicMock
    ) -> None:
        """The pre-drain. A disabled archive answers "off" without any IO, and
        flashing "collecting…" at it for a full tick would be a lie."""
        mock_ctx.guild.voice_client = None
        embed = await _skeleton(mock_ctx, self._operator())
        postgres = next(f.value or "" for f in embed.fields if f.name == "Postgres")
        assert "collecting" not in postgres
        assert "archive disabled" in postgres

    async def test_the_instant_blocks_are_already_filled_in_the_skeleton(
        self, mock_ctx: MagicMock
    ) -> None:
        """Config and Discord need no IO, so making the user wait for them would be
        pure loss. Versions rides prepare()."""
        mock_ctx.guild.voice_client = None
        embed = await _skeleton(mock_ctx, self._operator())
        rendered = {f.name: (f.value or "") for f in embed.fields}
        assert "ENVIRONMENT" in rendered["Config"]
        assert "shards" in rendered["Discord"]
        assert "collecting" not in rendered["Versions"]

    async def test_the_send_happens_before_a_slow_block_resolves(
        self, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The real window would add 2s of pure wall time after the gate opens.
        monkeypatch.setattr(debug, "_PG_WINDOW_SECS", 0.0)
        mock_ctx.guild.voice_client = None
        gate = asyncio.Event()
        message = MagicMock(spec=discord.Message)
        message.edit = AsyncMock()
        mock_ctx.channel.send = AsyncMock(return_value=message)

        async def slow_stats() -> Any:
            await gate.wait()
            return _archive_stats()

        archive = MagicMock()
        archive.stats = slow_stats
        task = asyncio.create_task(
            run_debug_dashboard(
                mock_ctx, self._operator(archive=archive, archive_enabled=True)
            )
        )
        async with asyncio.timeout(3):
            while mock_ctx.channel.send.await_count == 0:
                await asyncio.sleep(0)
            # Sent, and Postgres is still a placeholder — the contract.
            call = mock_ctx.channel.send.await_args
            assert call is not None
            sent = call.kwargs["embeds"][0]
            assert "collecting" in next(
                f.value or "" for f in sent.fields if f.name == "Postgres"
            )
            gate.set()
            await task

    async def test_a_block_that_misses_the_deadline_says_so(
        self, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The improvement over one all-or-nothing timeout: a hung dependency costs
        its own block and nothing else."""
        mock_ctx.guild.voice_client = None
        monkeypatch.setattr(debug, "DEBUG_DEADLINE_SECS", 0.3)
        monkeypatch.setattr(debug, "DEBUG_TICK_SECS", 0.01)
        # The sampler owns a real window; shrink it so only the hung probe is
        # slow enough to miss the deadline.
        monkeypatch.setattr(debug, "_CPU_WINDOW_SECS", 0.0)

        async def never() -> Any:
            await asyncio.Event().wait()

        archive = MagicMock()
        archive.stats = never
        embed = await _snapshot(
            mock_ctx, self._operator(archive=archive, archive_enabled=True)
        )
        rendered = {f.name: (f.value or "") for f in embed.fields}
        assert "timed out" in rendered["Postgres"]
        # Every other block still landed — that is the whole point.
        assert "timed out" not in rendered["Runtime"]
        assert "uptime" in rendered["Runtime"]

    async def test_a_probe_that_raises_degrades_to_its_own_blocks(
        self, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild.voice_client = None
        with patch.object(debug, "_dbsize", side_effect=RuntimeError("boom")):
            embed = await _snapshot(mock_ctx, self._operator())
        rendered = {f.name: (f.value or "") for f in embed.fields}
        for name in debug._PROBE_BLOCKS["redis"]:
            assert "unavailable (RuntimeError)" in rendered[name], name
        assert "version" in rendered["Build"]  # a different probe, unaffected

    async def test_a_dead_redis_does_not_blank_the_runtime_block(
        self, mock_ctx: MagicMock
    ) -> None:
        """Regression: Runtime reads os.times/proc/pool_state and needs no Redis at
        all, but it used to share a probe whose FIRST await was an unbounded Redis
        read. A Redis that accepted the socket and never answered hid uptime, memory,
        task count and pool state, and the block then claimed IT had timed out — in
        exactly the state -debug exists to diagnose.
        """

        async def never(*_a: object, **_k: object) -> None:
            await asyncio.Event().wait()

        mock_ctx.guild.voice_client = None
        redis = MagicMock()
        redis.info = never
        redis.dbsize = never
        with patch.object(debug, "_REDIS_PROBE_TIMEOUT_SECS", 0.05):
            embed = await _snapshot(mock_ctx, self._operator(redis=redis))
        rendered = {f.name: (f.value or "") for f in embed.fields}
        assert "uptime" in rendered["Runtime"]
        assert "timed out" not in rendered["Runtime"]
        assert "unavailable" not in rendered["Runtime"]

    async def test_a_non_operator_needs_no_loop_at_all(
        self, mock_ctx: MagicMock
    ) -> None:
        """Nothing a non-operator sees costs IO, so the driver degrades to one send
        with no edits — no placeholders, no deadline wait."""
        mock_ctx.guild.voice_client = None
        message = MagicMock(spec=discord.Message)
        message.edit = AsyncMock()
        mock_ctx.channel.send = AsyncMock(return_value=message)
        await run_debug_dashboard(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        mock_ctx.channel.send.assert_awaited_once()
        message.edit.assert_not_awaited()
        call = mock_ctx.channel.send.await_args
        assert call is not None
        sent = call.kwargs["embeds"][0]
        assert "collecting" not in str(sent.to_dict())

    async def test_it_stays_off_the_now_playing_host(self, mock_ctx: MagicMock) -> None:
        """channel.send, not ctx.send: an edited message must not be the NP host,
        whose progress updater re-renders it every few seconds."""
        mock_ctx.guild.voice_client = None
        await _snapshot(mock_ctx, self._operator())
        mock_ctx.send.assert_not_awaited()
        mock_ctx.channel.send.assert_awaited_once()

    async def test_git_sha_is_resolved_off_the_event_loop(
        self, mock_ctx: MagicMock
    ) -> None:
        """git_sha() shells out; on the loop it would stall the skeleton send it is
        supposed to be racing."""
        mock_ctx.guild.voice_client = None
        seen: list[str] = []
        real_run_in_executor = asyncio.get_running_loop().run_in_executor

        async def spy(executor: Any, fn: Any, *args: Any) -> Any:
            seen.append(getattr(fn, "__name__", ""))
            return await real_run_in_executor(executor, fn, *args)

        with patch.object(
            asyncio.get_running_loop(), "run_in_executor", side_effect=spy
        ):
            await _snapshot(mock_ctx, self._operator())
        assert "git_sha" in seen


class TestTraceIdInTheFooter:
    """The trace id is the whole point of debug mode — the help text says "paste
    that id to the operator". Nothing else in the suite renders one: conftest
    installs no TracerProvider, so `trace_id_of()` answers "" everywhere and every
    trace assertion elsewhere holds vacuously. These build a real span context.
    """

    TRACE_ID = 0x4BF92F3577B34DA6A3CE929D0E0E4736
    HEX = "4bf92f3577b34da6a3ce929d0e0e4736"

    @classmethod
    def _span(cls) -> trace_api.Span:
        return trace_api.NonRecordingSpan(
            trace_api.SpanContext(
                trace_id=cls.TRACE_ID,
                span_id=0x00F067AA0BA902B7,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED),
            )
        )

    def test_the_footer_carries_the_trace_id(self) -> None:
        footer = debug.debug_footer(span=self._span(), elapsed_ms=12.0)
        assert f"trace {self.HEX}" in footer

    def test_an_embed_gets_the_trace_id(self) -> None:
        embed = discord.Embed(description="x")
        debug.decorate_embeds([embed], span=self._span(), shard_id=0)
        assert self.HEX in (embed.footer.text or "")

    def test_an_error_embeds_existing_trace_is_not_repeated(self) -> None:
        """_command_error already puts `trace: <id>` on error embeds. Twice in one
        footer reads as two different traces."""
        embed = discord.Embed(description="x")
        embed.set_footer(text=f"trace: {self.HEX}")
        debug.decorate_embeds([embed], span=self._span(), elapsed_ms=5.0)
        assert (embed.footer.text or "").count(self.HEX) == 1

    def test_skip_trace_is_what_suppresses_it(self) -> None:
        """Pins the guard itself, not just its effect: flipping skip_trace has to
        be the thing that changes the output."""
        span = self._span()
        assert self.HEX in debug.debug_footer(span=span, skip_trace=False)
        assert self.HEX not in debug.debug_footer(span=span, skip_trace=True)

    async def test_the_snapshot_footer_carries_it_too(
        self, mock_ctx: MagicMock
    ) -> None:
        with patch.object(trace_api, "get_current_span", return_value=self._span()):
            embed = await _snapshot(
                mock_ctx,
                DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
            )
        assert self.HEX in (embed.footer.text or "")


class TestDecorationIsIdempotent:
    """decorate_embeds REPLACES a previous debug suffix instead of appending after
    it. The one cached embed sent more than once (play_message, re-served by -now
    during the crash-recovery window) used to grow a fresh suffix on every send."""

    def test_redecorating_replaces_the_suffix(self) -> None:
        embed = discord.Embed(title="x")
        debug.decorate_embeds([embed], elapsed_ms=5.0, shard_id=0)
        debug.decorate_embeds([embed], elapsed_ms=9.0, shard_id=0)
        text = embed.footer.text or ""
        assert text.count(debug._DEBUG_MARK) == 1
        assert "9 ms" in text
        assert "5 ms" not in text

    def test_an_original_footer_survives_redecoration(self) -> None:
        embed = discord.Embed(title="x")
        embed.set_footer(text="Avg Bitrate: 128 kbps")
        debug.decorate_embeds([embed], shard_id=1)
        debug.decorate_embeds([embed], shard_id=2)
        text = embed.footer.text or ""
        assert text.startswith("Avg Bitrate: 128 kbps · ")
        assert text.count("Avg Bitrate") == 1
        assert text.count(debug._DEBUG_MARK) == 1
        assert "shard 2" in text
        assert "shard 1" not in text

    def test_our_own_trace_does_not_suppress_the_fresh_one(self) -> None:
        """The old check read the whole footer, so our own previous suffix's trace
        segment suppressed its replacement."""
        span = TestTraceIdInTheFooter._span()
        embed = discord.Embed(title="x")
        debug.decorate_embeds([embed], span=span, shard_id=0)
        debug.decorate_embeds([embed], span=span, shard_id=0)
        assert (embed.footer.text or "").count(TestTraceIdInTheFooter.HEX) == 1

    def test_an_error_embeds_own_trace_still_wins(self) -> None:
        """skip_trace is decided from the pre-suffix footer: _command_error's
        `trace:` line keeps suppressing ours across any number of decorations."""
        embed = discord.Embed(title="x")
        embed.set_footer(text=f"trace: {TestTraceIdInTheFooter.HEX}")
        span = TestTraceIdInTheFooter._span()
        debug.decorate_embeds([embed], span=span, elapsed_ms=1.0)
        debug.decorate_embeds([embed], span=span, elapsed_ms=2.0)
        assert (embed.footer.text or "").count(TestTraceIdInTheFooter.HEX) == 1

    def test_nothing_to_show_strips_a_stale_suffix(self) -> None:
        embed = discord.Embed(title="x")
        debug.decorate_embeds([embed], shard_id=0)
        assert embed.footer.text is not None
        debug.decorate_embeds([embed])
        assert embed.footer.text is None

    def test_nothing_to_show_restores_the_original_footer(self) -> None:
        embed = discord.Embed(title="x")
        embed.set_footer(text="environment: test")
        debug.decorate_embeds([embed], shard_id=0)
        debug.decorate_embeds([embed])
        assert embed.footer.text == "environment: test"

    def test_a_doubled_legacy_suffix_is_fully_replaced(self) -> None:
        """Pre-idempotency footers can carry two suffixes; the first mark is the
        boundary, so one decoration heals the whole tail."""
        embed = discord.Embed(title="x")
        embed.set_footer(
            text=f"base · {debug._DEBUG_MARK} 3 ms · {debug._DEBUG_MARK} 4 ms"
        )
        debug.decorate_embeds([embed], shard_id=0)
        assert embed.footer.text == f"base · {debug._DEBUG_MARK} shard 0"

    def test_an_undecorated_footer_is_left_exactly_alone(self) -> None:
        """Nothing to add and nothing stale to strip. Reachable: debug on, in a DM
        (no shard), before the sampler's first tick, outside a command span."""
        embed = discord.Embed(title="x")
        embed.set_footer(text="environment: test", icon_url="https://e/i.png")
        debug.decorate_embeds([embed])
        assert embed.footer.text == "environment: test"
        assert embed.footer.icon_url == "https://e/i.png"

    def test_an_empty_footer_stays_empty(self) -> None:
        embed = discord.Embed(title="x")
        debug.decorate_embeds([embed])
        assert embed.footer.text is None

    def test_a_footer_that_is_only_a_suffix_loses_its_icon_too(self) -> None:
        """Discord rejects an icon with no text, so stripping the text takes it."""
        embed = discord.Embed(title="x")
        embed.set_footer(
            text=f"{debug._DEBUG_MARK} shard 0", icon_url="https://e/i.png"
        )
        debug.decorate_embeds([embed])
        assert embed.to_dict().get("footer") in (None, {})

    def test_a_near_limit_footer_keeps_accepting_a_fresh_suffix(self) -> None:
        """The clip falls on the base. Truncating the join cuts the ` · 🐞 `
        boundary off, after which no suffix is ever accepted again."""
        embed = discord.Embed(title="x")
        embed.set_footer(text="B" * (debug.FOOTER_LIMIT - 4))
        debug.decorate_embeds([embed], elapsed_ms=5.0, shard_id=0)
        assert "5 ms" in (embed.footer.text or "")
        debug.decorate_embeds([embed], elapsed_ms=9.0, shard_id=0)
        text = embed.footer.text or ""
        assert text.count(debug._DEBUG_MARK) == 1
        assert "9 ms" in text and "5 ms" not in text
        assert len(text) <= debug.FOOTER_LIMIT

    def test_a_mark_that_is_not_a_suffix_takes_the_tail_with_it(self) -> None:
        """The first mark is the boundary, so anything after it is discarded. Safe
        only while no bot-authored footer contains one — asserted below."""
        embed = discord.Embed(title="x")
        embed.set_footer(text=f"a · {debug._DEBUG_MARK} b · c")
        debug.decorate_embeds([embed], shard_id=0)
        assert embed.footer.text == f"a · {debug._DEBUG_MARK} shard 0"

    def test_no_footer_the_bot_writes_contains_the_mark(self) -> None:
        """The invariant the test above depends on. `-debug`'s own mark is in its
        title, and that card never passes through decorate_embeds."""
        import re

        source = Path("src").rglob("*.py")
        offenders: list[str] = []
        for path in source:
            text = path.read_text(encoding="utf-8")
            for match in re.finditer(r"set_footer\((.{0,200}?)\)", text, re.S):
                if debug._DEBUG_MARK in match.group(1):
                    offenders.append(f"{path}: {match.group(1)[:60]}")
        assert not offenders, offenders


class TestOperatorGate:
    """`-debug` is reachable by any user in any guild, and by DM. Everything that
    describes the HOST rather than the caller's own server is owner-only."""

    _HOST_BLOCKS = ("Build", "Config", "Runtime", "Discord", "Redis", "Postgres",
                    "Checks")  # fmt: skip

    async def test_non_operator_sees_only_versions_and_their_own_server(
        self, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild.voice_client = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=3),
        )
        assert [f.name for f in embed.fields] == ["Versions", "This server"]

    async def test_non_operator_never_reaches_a_host_collector(
        self, mock_ctx: MagicMock
    ) -> None:
        """Not just absent from the embed — never collected. A gate that renders
        the block and then drops it still reads the secrets, and still pays the IO."""
        mock_ctx.guild.voice_client = None
        with (
            patch.object(debug, "config_lines") as config,
            patch.object(debug, "runtime_lines") as runtime,
            patch.object(debug, "redis_lines") as redis_block,
            patch.object(debug, "build_lines") as build,
            patch.object(debug, "checks_lines") as checks,
        ):
            await _snapshot(
                mock_ctx,
                DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
            )
        for collector in (config, runtime, redis_block, build, checks):
            collector.assert_not_called()

    async def test_a_non_operators_snapshot_carries_no_configured_value(
        self, mock_ctx: MagicMock
    ) -> None:
        """The disclosure this gate exists for: internal hostnames and ports. The
        sentinel stands in for every URL-kind row."""
        mock_ctx.guild.voice_client = None
        with patch.object(
            debug, "config_lines", return_value=["REDIS_URL  redis://sentinel:6379"]
        ):
            embed = await _snapshot(
                mock_ctx,
                DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
            )
        assert "sentinel" not in str(embed.to_dict())

    async def test_a_non_operator_is_told_why_rather_than_left_guessing(
        self, mock_ctx: MagicMock
    ) -> None:
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert embed.description is not None
        assert "bot owner" in embed.description
        assert "-ping" in embed.description

    async def test_the_operator_notice_is_absent_for_an_operator(
        self, mock_ctx: MagicMock
    ) -> None:
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(
                debug_enabled=False,
                debug_overridden=False,
                players=0,
                operator=True,
            ),
        )
        assert embed.description is not None
        assert "bot owner" not in embed.description

    @pytest.mark.parametrize("block", _HOST_BLOCKS)
    async def test_every_host_block_is_operator_only(
        self, mock_ctx: MagicMock, block: str
    ) -> None:
        """Parametrized so a block added later must decide which side it is on:
        appending it to the operator branch without adding it here fails nothing,
        but adding it to the PUBLIC branch fails this."""
        mock_ctx.guild.voice_client = None
        public = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert block not in [f.name for f in public.fields]


class TestSnapshotEmbed:
    async def test_renders_every_block_in_a_guild(self, mock_ctx: MagicMock) -> None:
        mock_ctx.guild.voice_client = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(
                debug_enabled=True, debug_overridden=True, players=3, operator=True
            ),
        )
        names = [f.name for f in embed.fields]
        # Config outgrew Discord's 1024-char field cap once it carried -debug's own
        # loop tunables, so _codeblock_fields splits it — the continuation keeps the
        # block's position rather than moving to the end.
        assert names == [
            "Build",
            "Versions",
            "Config",
            "Config (cont.)",
            "Runtime",
            "Discord",
            "Redis",
            "Postgres",
            "This server",
            "Checks",
        ]
        assert embed.description is not None
        assert "**on**" in embed.description
        assert "(saved here)" in embed.description

    async def test_dm_omits_the_guild_block(self, mock_ctx: MagicMock) -> None:
        mock_ctx.guild = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert "This server" not in [f.name for f in embed.fields]
        assert embed.description is not None
        assert "direct messages" in embed.description
        assert "(host default)" in embed.description

    async def test_no_credential_reaches_the_embed(
        self, mock_ctx: MagicMock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """End-to-end guard on the whole rendered surface, not just the row
        renderer: the embed is built to be pasted into an issue.

        `operator=True` and a guild are load-bearing, not incidental. Config is the
        only block that reads any of these three variables and it sits behind the
        operator gate, so the earlier non-operator DM version of this test asserted
        three credentials were absent from a card that never read them — which is
        why the render of Config is asserted first.
        """
        mock_ctx.guild.voice_client = None
        monkeypatch.setenv("DISCORD_TOKEN", "tok-should-never-render")
        monkeypatch.setenv(
            "POSTGRES_URL", "postgresql://bot:pw-secret@db:5432/musicbot"
        )
        monkeypatch.setenv("REDIS_URL", "redis://cacheuser:cachepw@cache:6379")
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(
                debug_enabled=False, debug_overridden=False, players=0, operator=True
            ),
        )
        names = [f.name for f in embed.fields]
        assert "Config" in names and "This server" in names
        # to_dict(), not the field values: description and footer are rendered
        # surface too.
        rendered = str(embed.to_dict())
        for leaked in ("tok-should-never-render", "pw-secret", "cachepw"):
            assert leaked not in rendered

    async def test_footer_carries_environment(self, mock_ctx: MagicMock) -> None:
        mock_ctx.guild = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert embed.footer.text is not None
        assert embed.footer.text.startswith("environment: ")

    async def test_footer_carries_the_debug_suffix_when_given_one(
        self, mock_ctx: MagicMock
    ) -> None:
        """-debug is an edit loop replying through channel.send, so neither decoration
        seam reaches it — the cog threads a pre-rendered suffix in instead."""
        mock_ctx.guild = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(
                debug_enabled=True,
                debug_overridden=False,
                players=0,
                debug_suffix="🐞 shard 0 · cpu 9%",
            ),
        )
        footer = embed.footer.text or ""
        assert footer.startswith("environment: ")
        assert footer.endswith("🐞 shard 0 · cpu 9%")

    async def test_footer_has_no_suffix_by_default(self, mock_ctx: MagicMock) -> None:
        mock_ctx.guild = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert "🐞" not in (embed.footer.text or "")


class TestDebugCardCarriesTheSuffixEndToEnd:
    """The cog→card wiring, driven through the real command. TestSnapshotEmbed above
    hand-builds a DebugInputs, so it pins only the renderer's concatenation."""

    @staticmethod
    def _enable(cog: MusicBotCog, ctx: MagicMock) -> None:
        cog.debug_settings._overrides[ctx.guild.id] = True
        cog.debug_settings._sampler._snapshot = debug.RuntimeSnapshot(
            cpu_percent=9.0, mem_percent=8.0, lag_ms=1.5, tasks=3, pool_workers=4
        )

    @staticmethod
    async def _run(cog: MusicBotCog, ctx: MagicMock) -> discord.Embed:
        """Drive the real command and return the finished card."""
        message = MagicMock(spec=discord.Message)
        message.edit = AsyncMock()
        ctx.channel.send = AsyncMock(return_value=message)
        await command_callback(MusicBotCog.debug)(cog, ctx)
        call = message.edit.await_args or ctx.channel.send.await_args
        assert call is not None
        return cast(discord.Embed, call.kwargs["embeds"][0])

    async def test_the_owners_card_carries_the_footer(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        self._enable(music_bot, mock_ctx)
        mock_ctx.guild.shard_id = 2
        footer = (await self._run(music_bot, mock_ctx)).footer.text or ""
        assert "🐞" in footer and "shard 2" in footer
        assert "cpu 9%" in footer  # the owner does get the host metrics

    async def test_a_non_owner_gets_the_footer_without_host_metrics(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """The card withholds its Runtime block from a non-owner and says so in the
        same embed, so the footer must not print those figures."""
        self._enable(music_bot, mock_ctx)
        mock_ctx.guild.shard_id = 2
        mock_ctx.bot.is_owner = AsyncMock(return_value=False)
        embed = await self._run(music_bot, mock_ctx)
        footer = embed.footer.text or ""
        assert "🐞" in footer and "shard 2" in footer
        for withheld in ("cpu ", "mem ", "lag ", "tasks ", "pool "):
            assert withheld not in footer
        # The notice that makes this necessary is really on the card.
        assert "bot owner only" in (embed.description or "")

    async def test_no_footer_when_the_guild_has_debug_off(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        embed = await self._run(music_bot, mock_ctx)
        assert "🐞" not in (embed.footer.text or "")


class TestDebugSuffixBuilder:
    """The cog's half: what the two live dashboards are handed."""

    def test_none_while_the_guild_has_debug_off(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        assert music_bot._debug_suffix(mock_ctx) is None

    def test_carries_shard_and_runtime_but_never_a_trace(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """No trace because no span is passed; the `skip_trace=True` alongside is
        for a future caller that does. Asserted either way — both cards already
        print `trace: <id>` themselves."""
        music_bot.debug_settings._overrides[mock_ctx.guild.id] = True
        mock_ctx.guild.shard_id = 4
        music_bot.debug_settings._sampler._snapshot = debug.RuntimeSnapshot(
            cpu_percent=9.0, mem_percent=8.0, lag_ms=1.5, tasks=3, pool_workers=4
        )
        with trace_api.use_span(TestTraceIdInTheFooter._span(), end_on_exit=False):
            suffix = music_bot._debug_suffix(mock_ctx)
        assert suffix is not None
        assert "shard 4" in suffix and "cpu 9%" in suffix
        assert "trace" not in suffix

    def test_in_a_dm_it_falls_back_to_the_host_default_and_omits_the_shard(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """Both are conditional expressions, so coverage reports no partial branch
        and this gap is invisible in the numbers. -debug is DM-reachable."""
        mock_ctx.guild = None
        music_bot.debug_settings._default = True
        music_bot.debug_settings._sampler._snapshot = debug.RuntimeSnapshot(
            cpu_percent=9.0, mem_percent=8.0, lag_ms=1.5, tasks=3, pool_workers=4
        )
        suffix = music_bot._debug_suffix(mock_ctx) or ""
        assert "cpu 9%" in suffix
        assert "shard" not in suffix

    def test_a_dm_with_nothing_to_report_yields_none_rather_than_a_bare_mark(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """No guild and no snapshot leaves debug_footer with nothing to say; `or
        None` stops a lone 🐞 reaching the card."""
        mock_ctx.guild = None
        music_bot.debug_settings._default = True
        assert music_bot._debug_suffix(mock_ctx) is None

    def test_the_first_segment_is_the_shard_not_an_elapsed_time(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """The dashboards only edit when the render differs, and elapsed-ms — which
        debug_footer puts first — would differ every tick."""
        music_bot.debug_settings._overrides[mock_ctx.guild.id] = True
        mock_ctx.guild.shard_id = 0
        suffix = music_bot._debug_suffix(mock_ctx) or ""
        assert suffix.removeprefix("🐞 ").split(" · ")[0] == "shard 0"


class TestCommandRegistration:
    def test_registered_under_utility_with_self_documenting_extras(self) -> None:
        from src.help import CATEGORY_COMMANDS
        from src.musicbot import MusicBot

        command = MusicBot.debug
        assert isinstance(command, commands.Command)
        assert command.name == "debug"
        assert "dbg" in command.aliases
        assert command.extras["category"] == "Utility"
        assert command.extras["examples"]
        assert "debug" in CATEGORY_COMMANDS["Utility"]


class TestCgroupReaders:
    """The container-metrics readers, against fixture files: the real ones exist
    only on Linux, and the dev machine here is macOS."""

    @pytest.fixture
    def cgroup(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
        monkeypatch.setattr(debug, "_CGROUP", tmp_path)
        return tmp_path

    @pytest.fixture
    def in_container(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Pinned in both directions, never left to the host: this suite runs on
        macOS AND inside a Linux container (`just container-test`), so an unpatched
        _in_container() steers the branch differently in the two."""
        monkeypatch.setattr(debug, "_in_container", lambda: True)

    def test_cpu_sample_prefers_the_container_wide_reading(
        self, cgroup: Path, in_container: None
    ) -> None:
        """cgroup cpu.stat covers FFmpeg subprocesses and pool workers; os.times()
        counts children only once REAPED, so FFmpeg would land at song end."""
        (cgroup / "cpu.stat").write_text("usage_usec 4500000\nuser_usec 3000000\n")
        sample = debug.read_cpu_sample()
        assert sample.scope == "container"
        assert sample.seconds == 4.5

    def test_a_bare_metal_host_is_not_read_as_a_container(
        self, cgroup: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """cpu.stat is readable off-container too, where /sys/fs/cgroup is the ROOT
        cgroup: its counter is EVERY process on the machine. Ungated, the bot reports
        the whole host's CPU as its own — a wrong number, not a missing one."""
        monkeypatch.setattr(debug, "_in_container", lambda: False)
        (cgroup / "cpu.stat").write_text("usage_usec 4500000\nuser_usec 3000000\n")
        sample = debug.read_cpu_sample()
        assert sample.scope == "process"
        assert sample.seconds != 4.5

    def test_cpu_sample_falls_back_to_process_times(
        self, cgroup: Path, in_container: None
    ) -> None:
        sample = debug.read_cpu_sample()
        assert sample.scope == "process"
        assert sample.seconds >= 0

    def test_malformed_cpu_stat_does_not_raise(
        self, cgroup: Path, in_container: None
    ) -> None:
        (cgroup / "cpu.stat").write_text("usage_usec not-a-number\n")
        assert debug.read_cpu_sample().scope == "process"

    def test_cores_honour_the_quota(self, cgroup: Path) -> None:
        """A container capped at 1.5 cores must not report 12% while saturated."""
        (cgroup / "cpu.max").write_text("150000 100000")
        assert debug.cpu_cores() == 1.5

    def test_unlimited_quota_uses_the_host_count(self, cgroup: Path) -> None:
        (cgroup / "cpu.max").write_text("max 100000")
        assert debug.cpu_cores() == float(os.cpu_count() or 1)

    def test_memory_prefers_the_cgroup_pair(
        self, cgroup: Path, in_container: None
    ) -> None:
        (cgroup / "memory.current").write_text("536870912")
        (cgroup / "memory.max").write_text("1073741824")
        reading = debug.read_memory()
        assert reading.scope == "container"
        assert reading.percent == 50.0

    def test_uncapped_memory_falls_back_to_host_total(
        self,
        cgroup: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        in_container: None,
    ) -> None:
        (cgroup / "memory.current").write_text("1048576")
        (cgroup / "memory.max").write_text("max")
        meminfo = tmp_path / "meminfo"
        meminfo.write_text("MemTotal:       2048 kB\nMemFree:  1024 kB\n")
        monkeypatch.setattr(debug, "_PROC_MEMINFO", meminfo)
        assert debug.read_memory().limit_bytes == 2048 * 1024

    def test_a_bare_metal_host_does_not_report_the_whole_machine(
        self, cgroup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """memory.current's twin of the cpu.stat bug, and the worse of the two: it
        is an absolute number rather than a rate, so the whole machine's usage
        labelled `(container)` looks entirely authoritative."""
        monkeypatch.setattr(debug, "_in_container", lambda: False)
        (cgroup / "memory.current").write_text("536870912")
        (cgroup / "memory.max").write_text("1073741824")
        status = tmp_path / "status"
        status.write_text("Name:\tpython\nVmRSS:\t  262144 kB\n")
        monkeypatch.setattr(debug, "_PROC_STATUS", status)
        reading = debug.read_memory()
        assert reading.scope == "process"
        assert reading.used_bytes == 262144 * 1024

    def test_proc_rss_is_the_second_choice(
        self, cgroup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        status = tmp_path / "status"
        status.write_text("Name:\tpython\nVmRSS:\t  262144 kB\n")
        monkeypatch.setattr(debug, "_PROC_STATUS", status)
        reading = debug.read_memory()
        assert reading.scope == "process"
        assert reading.label == "rss"
        assert reading.used_bytes == 262144 * 1024

    def test_no_proc_at_all_still_reports_something(
        self, cgroup: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """macOS dev: ru_maxrss is a PEAK, and its units differ by platform —
        bytes on darwin, KiB on Linux. It must be labeled as a peak, not an rss.

        /proc is pointed at a nonexistent path rather than left alone: this suite
        runs on macOS AND inside a Linux container (`just container-test`), where
        the real file exists and the fallback would never be reached.
        """
        monkeypatch.setattr(debug, "_PROC_STATUS", tmp_path / "absent")
        reading = debug.read_memory()
        assert reading.label == "peak"
        assert reading.used_bytes > 0

    @pytest.mark.parametrize(
        ("platform", "expected_bytes"), [("darwin", 4096), ("linux", 4096 * 1024)]
    )
    def test_ru_maxrss_units_are_platform_specific(
        self,
        cgroup: Path,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        platform: str,
        expected_bytes: int,
    ) -> None:
        """getrusage(2): ru_maxrss is BYTES on darwin and KiB on Linux. `> 0` holds
        with the conversion inverted, and the error is a factor of 1024 in the one
        memory number a macOS dev run can produce."""
        monkeypatch.setattr(debug, "_PROC_STATUS", tmp_path / "absent")
        monkeypatch.setattr(debug.sys, "platform", platform)
        monkeypatch.setattr(
            debug.resource,
            "getrusage",
            lambda _who: SimpleNamespace(ru_maxrss=4096),
        )
        reading = debug.read_memory()
        assert reading.label == "peak"
        assert reading.used_bytes == expected_bytes

    def test_percent_is_none_without_a_ceiling(self) -> None:
        reading = debug.MemoryReading(
            used_bytes=100, limit_bytes=None, scope="process", label="rss"
        )
        assert reading.percent is None


class TestCpuPercent:
    def test_rate_over_the_window(self) -> None:
        first = debug.CpuSample(
            seconds=10.0, monotonic=100.0, scope="container", cores=4
        )
        second = debug.CpuSample(
            seconds=12.0, monotonic=101.0, scope="container", cores=4
        )
        # 2 cpu-seconds over 1 wall second across 4 cores = 50%.
        assert debug.cpu_percent(first, second) == 50.0

    def test_zero_window_is_none_not_a_division_error(self) -> None:
        sample = debug.CpuSample(seconds=1.0, monotonic=100.0, scope="process", cores=1)
        assert debug.cpu_percent(sample, sample) is None

    def test_counter_going_backwards_clamps_at_zero(self) -> None:
        first = debug.CpuSample(seconds=5.0, monotonic=1.0, scope="process", cores=1)
        second = debug.CpuSample(seconds=4.0, monotonic=2.0, scope="process", cores=1)
        assert debug.cpu_percent(first, second) == 0.0

    def test_the_sampling_window_is_never_zero(self) -> None:
        """CPU% is a rate and exists only ACROSS a window. Shrinking this constant to
        0.0 looks like removing a half-second of latency and nothing in the suite
        goes red, but both CPU rows then divide over scheduling noise — and it is the
        loop-lag measurement too, which becomes a reading of nothing."""
        assert debug._CPU_WINDOW_SECS > 0


class TestRuntimeBlock:
    async def test_uptime_is_a_plain_duration_not_discord_markup(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every row in this block is rendered inside a ``` fence, and Discord prints
        markup there as its literal source: the row read `uptime <t:1785881580:R>` to
        every operator who ran it, rather than "2 hours ago"."""
        monkeypatch.setattr(debug, "_PROCESS_START", time.time() - 3725)
        line = next(line for line in debug.runtime_lines() if line.startswith("uptime"))
        assert "<t:" not in line
        assert line == "uptime       1:02:05"


class TestGitSha:
    @pytest.fixture(autouse=True)
    def _clear_cache(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(debug, "_git_sha_cache", None)

    def test_baked_env_wins(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The deployed case: GIT_SHA is an ENV in the runtime image, because an
        OCI label is invisible from inside the container."""
        monkeypatch.setenv("GIT_SHA", "abc1234-dirty.deadbeef")
        assert debug.git_sha() == "abc1234-dirty.deadbeef"

    def test_falls_back_to_the_checkout(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """`just run` from a checkout has no image and therefore no baked SHA.

        The subprocess is faked rather than really run: the runtime image copies
        src/ and pyproject.toml but no .git, so a real `git rev-parse` succeeds
        under `just test` and fails under `just container-test`.
        """
        monkeypatch.delenv("GIT_SHA", raising=False)
        monkeypatch.setattr(
            debug.subprocess,
            "run",
            MagicMock(return_value=SimpleNamespace(stdout="a1b2c3d\n")),
        )
        assert debug.git_sha() == "a1b2c3d"

    def test_real_checkout_answers_when_there_is_one(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unfaked, so the argv itself is exercised — but skipped where there is
        no checkout, which is exactly the runtime image."""
        monkeypatch.delenv("GIT_SHA", raising=False)
        if not Path(__file__).resolve().parent.parent.joinpath(".git").exists():
            pytest.skip("no git checkout (runtime image)")
        assert re.fullmatch(r"[0-9a-f]{7,40}", debug.git_sha())

    def test_no_env_and_no_git_is_unknown(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("GIT_SHA", raising=False)
        monkeypatch.setattr(
            debug.subprocess, "run", MagicMock(side_effect=OSError("no git"))
        )
        assert debug.git_sha() == "unknown"

    def test_cached_for_process_lifetime(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GIT_SHA", "first")
        assert debug.git_sha() == "first"
        monkeypatch.setenv("GIT_SHA", "second")
        assert debug.git_sha() == "first"


def _redis_info(**overrides: object) -> dict[str, object]:
    """A realistic INFO reply. fakeredis does not implement INFO at all — it raises
    ResponseError — so every Redis-block test supplies its own."""
    info: dict[str, object] = {
        "used_cpu_sys": 1.0,
        "used_cpu_user": 2.0,
        "used_memory": 44040192,
        "maxmemory": 268435456,
        "mem_fragmentation_ratio": 1.08,
        "maxmemory_policy": "volatile-lru",
        "connected_clients": 3,
        "instantaneous_ops_per_sec": 148,
        "keyspace_hits": 870,
        "keyspace_misses": 130,
        "evicted_keys": 0,
        "rdb_last_bgsave_status": "ok",
        "aof_last_write_status": "ok",
        "db0": {"keys": 1284, "expires": 1192, "avg_ttl": 0},
    }
    info.update(overrides)
    return info


class TestRedisBlock:
    def _samples(
        self, **overrides: object
    ) -> tuple[debug.RedisSample, debug.RedisSample]:
        first = debug.RedisSample(info=_redis_info(**overrides), monotonic=100.0)
        second = debug.RedisSample(
            info=_redis_info(used_cpu_sys=1.005, used_cpu_user=2.005, **overrides),
            monotonic=101.0,
        )
        return first, second

    def test_renders_cpu_before_mem(self) -> None:
        first, second = self._samples()
        lines = debug.redis_lines(first, second, dbsize=1284)
        assert lines[0].startswith("cpu")
        assert lines[1].startswith("mem")

    def test_cpu_is_a_rate_across_the_window(self) -> None:
        first, second = self._samples()
        # 0.01 cpu-seconds over 1 wall second = 1%. No core normalization: Redis
        # is effectively single-threaded, so 100% is one saturated core.
        assert "1.0%" in debug.redis_lines(first, second, dbsize=0)[0]

    def test_memory_percent_is_against_maxmemory(self) -> None:
        """maxmemory is the threshold volatile-lru acts at, so the % reads as the
        eviction runway rather than as host pressure."""
        first, second = self._samples()
        assert "42 MB / 256 MB (16%)" in debug.redis_lines(first, second, dbsize=0)[1]

    def test_no_maxmemory_renders_absolute_only(self) -> None:
        first, second = self._samples(maxmemory=0)
        assert "(no maxmemory)" in debug.redis_lines(first, second, dbsize=0)[1]

    def test_persistent_keys_are_total_minus_expires(self) -> None:
        """The no-TTL population is exactly the set golden rule 12 protects."""
        first, second = self._samples()
        keys = next(
            line
            for line in debug.redis_lines(first, second, dbsize=1284)
            if "keys" in line
        )
        assert "1284 total · 92 persistent" in keys

    def test_degrades_when_info_is_unavailable(self) -> None:
        lines = debug.redis_lines(None, None, dbsize=None)
        assert len(lines) == 1
        assert "unavailable" in lines[0]

    async def test_read_sample_swallows_a_dead_redis(self) -> None:
        redis = MagicMock()
        redis.info = AsyncMock(side_effect=ConnectionError("down"))
        assert await debug.read_redis_sample(redis) is None

    async def test_read_sample_of_none_redis_is_none(self) -> None:
        assert await debug.read_redis_sample(None) is None

    async def test_read_sample_stamps_the_clock_it_was_taken_at(self) -> None:
        """The success path — untested until now, including its timestamp. Redis's
        CPU% is the gap between two of these, so a constant `monotonic` divides by a
        zero-or-fixed window and pins that row to garbage without failing anything.
        """
        redis = MagicMock()
        redis.info = AsyncMock(return_value=_redis_info())
        before = time.monotonic()
        sample = await debug.read_redis_sample(redis)
        assert sample is not None
        assert sample.info["maxmemory_policy"] == "volatile-lru"
        assert before <= sample.monotonic <= time.monotonic()

    async def test_dbsize_counts_the_keys(self, fake_redis: Redis) -> None:
        """The `keys` row's whole content. A hardcoded 0 or None reads as an
        empty/unknown keyspace on every host forever, which is exactly what a
        collector nobody executes looks like."""
        for i in range(3):
            await fake_redis.set(f"guild:{i}:state", "x")
        assert await debug._dbsize(fake_redis) == 3

    async def test_dbsize_without_redis_is_unknown(self) -> None:
        assert await debug._dbsize(None) is None

    async def test_dbsize_swallows_a_dead_redis(self) -> None:
        """A key count is a row, not a failure: it must not take the Redis block
        (or its sibling Checks) down with it."""
        redis = MagicMock()
        redis.dbsize = AsyncMock(side_effect=ConnectionError("down"))
        assert await debug._dbsize(redis) is None


class TestPostgresBlock:
    """postgres_lines as a pure renderer: instant fields from `after`, rates from
    the (before, after) bracket. The orchestration around it has its own class."""

    def test_renders_the_archive_stats(self) -> None:
        lines = "\n".join(debug.postgres_lines(None, None, _pg_stats()))
        assert "db 128 MB" in lines
        assert "play_history 96 MB" in lines
        assert "~412083 plays · 0 rejected" in lines
        assert "3 / 100" in lines
        assert "hit rate 99.4% (lifetime)" in lines

    def test_rates_come_from_the_window_deltas(self) -> None:
        before = _pg_stats()
        after = _pg_stats(
            active_backends=2,
            active_io_wait=1,
            active_time_ms=5_800.0,  # +800ms over a 2s window → 0.4 bk-s/s
            xacts_total=1_082,  # +82 → 41/s
            tuples_total=74_800,  # +24800 → 12.4k/s
            blks_hit=9_998,  # Δhit 998, Δread 2 → 99.8%
            blks_read=1_002,
            monotonic=102.0,
        )
        lines = "\n".join(debug.postgres_lines(None, before, after))
        assert "load         2 active (1 io-wait) · busy 0.4 bk-s/s" in lines
        assert "throughput   41 tx/s · 12.4k tuples/s" in lines
        assert "mem signal   window hit 99.8% · spill 0 B/s · deadlocks 0" in lines

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            # 0 active drops the parenthetical entirely.
            ({}, "load         0 active · busy"),
            # All on-CPU is information, not absence.
            ({"active_backends": 2}, "2 active (0 waiting) ·"),
            # One nonzero kind carries the -wait suffix.
            (
                {"active_backends": 2, "active_io_wait": 1},
                "2 active (1 io-wait) ·",
            ),
            (
                {"active_backends": 1, "active_other_wait": 1},
                "1 active (1 other-wait) ·",
            ),
            # Mixed kinds drop the suffix — the parenthetical is already
            # wait-scoped.
            (
                {
                    "active_backends": 3,
                    "active_io_wait": 1,
                    "active_lock_wait": 1,
                },
                "3 active (1 io · 1 lock) ·",
            ),
        ],
    )
    def test_wait_parenthetical_lists_only_nonzero_kinds(
        self, overrides: dict[str, Any], expected: str
    ) -> None:
        lines = "\n".join(debug.postgres_lines(None, None, _pg_stats(**overrides)))
        assert expected in lines

    def test_missing_first_sample_costs_only_the_rates(self) -> None:
        """The asymmetric failure contract: the instant half needs only `after`."""
        lines = debug.postgres_lines(None, None, _pg_stats(active_backends=1))
        joined = "\n".join(lines)
        assert "busy unknown" in joined
        assert "throughput   unknown" in joined
        assert "mem signal   unknown" in joined
        assert "1 active" in joined  # the instant half still renders
        assert "db 128 MB" in joined

    def test_zero_window_degrades_the_rates(self) -> None:
        """Two samples at the same monotonic instant have no window to rate over —
        `unknown`, never a ZeroDivisionError."""
        lines = "\n".join(
            debug.postgres_lines(None, _pg_stats(), _pg_stats(monotonic=100.0))
        )
        assert "busy unknown" in lines
        assert "throughput   unknown" in lines

    def test_idle_database_reads_as_idle_not_fault(self) -> None:
        before = _pg_stats()
        after = _pg_stats(monotonic=102.0)  # identical counters, real window
        lines = "\n".join(debug.postgres_lines(None, before, after))
        assert "load         0 active · busy 0.0 bk-s/s" in lines
        assert "throughput   0 tx/s · 0 tuples/s" in lines
        assert "mem signal   window hit n/a (idle) · spill 0 B/s · deadlocks 0" in lines

    def test_a_counter_reset_clamps_rates_at_zero(self) -> None:
        """pg_stat_reset() between samples makes every counter go backwards; the
        deltas clamp at 0 rather than rendering negative rates."""
        before = _pg_stats()
        after = _pg_stats(
            active_time_ms=100.0,
            xacts_total=10,
            tuples_total=500,
            blks_hit=90,
            blks_read=10,
            temp_bytes=0,
            monotonic=102.0,
        )
        lines = "\n".join(debug.postgres_lines(None, before, after))
        assert "busy 0.0 bk-s/s" in lines
        assert "throughput   0 tx/s · 0 tuples/s" in lines
        assert "window hit n/a (idle)" in lines
        assert "spill 0 B/s" in lines

    def test_row_order_is_pinned(self) -> None:
        container = debug.ContainerMetrics(
            cpu_percent=3.1, used_bytes=1048576, limit_bytes=2097152
        )
        lines = debug.postgres_lines(container, _pg_stats(), _pg_stats(monotonic=102.0))
        expected = (
            "cpu",
            "mem ",
            "load",
            "throughput",
            "mem signal",
            "storage",
            "rows",
            "conns",
            "cache",
        )
        assert len(lines) == len(expected)
        for line, prefix in zip(lines, expected, strict=True):
            assert line.startswith(prefix), line


class TestChecksBlock:
    async def _lines(
        self,
        *,
        info_overrides: Optional[dict[str, object]] = None,
        sample: object = "default",
        redis: object = "default",
        store: Optional[object] = None,
        archive_enabled: bool = True,
        default_password: Optional[bool] = None,
        depth: int = 0,
    ) -> str:
        if sample == "default":
            sample = debug.RedisSample(
                info=_redis_info(**(info_overrides or {})), monotonic=1.0
            )
        if redis == "default":
            redis = MagicMock()
        with patch.object(debug, "outbox_depth", AsyncMock(return_value=depth)):
            return "\n".join(
                await debug.checks_lines(
                    cast(Any, sample),
                    redis=cast(Any, redis),
                    store=cast(Any, store),
                    archive_enabled=archive_enabled,
                    default_password=default_password,
                )
            )

    async def test_correct_eviction_policy_passes(self) -> None:
        assert "✅ eviction" in await self._lines()

    async def test_wrong_eviction_policy_is_flagged(self) -> None:
        """Golden rule 12: allkeys-lru can evict history:outbox and the history
        lists, which is silent, unrecoverable data loss."""
        lines = await self._lines(info_overrides={"maxmemory_policy": "allkeys-lru"})
        assert "⚠️ eviction" in lines
        assert "must be volatile-lru" in lines

    async def test_failed_bgsave_is_flagged(self) -> None:
        """The early warning for the MISCONF incident: Redis keeps serving READS
        after a failed bgsave while refusing every write."""
        lines = await self._lines(info_overrides={"rdb_last_bgsave_status": "err"})
        assert "⚠️ persistence" in lines

    async def test_failed_aof_write_is_flagged(self) -> None:
        """The AOF half, which no test ever drove red. The compose Redis runs AOF,
        and the failure mode is silent in the safe-looking direction: a misspelled
        key defaults to "unknown", which this check reads as healthy, so the row
        goes permanently inert rather than permanently warning."""
        lines = await self._lines(info_overrides={"aof_last_write_status": "err"})
        assert "⚠️ persistence" in lines
        assert "aof err" in lines

    async def test_no_info_cannot_verify(self) -> None:
        assert "❔ redis" in await self._lines(sample=None)

    async def test_stranded_outbox_with_the_archive_off(self) -> None:
        lines = await self._lines(archive_enabled=False, depth=7)
        assert "⚠️ outbox" in lines
        assert "7 stranded" in lines

    async def test_outbox_depth_with_the_archive_on_is_not_an_alarm(self) -> None:
        lines = await self._lines(archive_enabled=True, depth=7)
        assert "✅ outbox" in lines
        assert "draining" in lines

    async def test_a_raising_outbox_helper_is_caught_here(self) -> None:
        """The outbox helpers RAISE by design (golden rule 5's split) because the
        drainer's backoff loop is their error handler — every other consumer owns
        its own try/except."""
        with patch.object(
            debug, "outbox_depth", AsyncMock(side_effect=RuntimeError("WRONGTYPE"))
        ):
            lines = "\n".join(
                await debug.checks_lines(
                    debug.RedisSample(info=_redis_info(), monotonic=1.0),
                    redis=cast(Any, MagicMock()),
                    store=None,
                    archive_enabled=True,
                    default_password=None,
                )
            )
        assert "❔ outbox" in lines
        assert "RuntimeError" in lines

    async def test_history_ttl_invariant(self) -> None:
        store = MagicMock()
        store.history_ttl = AsyncMock(return_value=-1)
        assert "✅ history ttl" in await self._lines(store=store)

    async def test_history_ttl_present_is_a_violation(self) -> None:
        """push_history PERSISTs that key unconditionally; a TTL there answers a
        guild with silence once it expires."""
        store = MagicMock()
        store.history_ttl = AsyncMock(return_value=3600)
        lines = await self._lines(store=store)
        assert "⚠️ history ttl" in lines
        assert "expires in 3600s" in lines

    async def test_history_ttl_unknown_when_redis_is_down(self) -> None:
        """@_guild_op swallows and returns the default, so None means "did not
        answer" — not "no TTL"."""
        store = MagicMock()
        store.history_ttl = AsyncMock(return_value=None)
        assert "❔ history ttl" in await self._lines(store=store)

    async def test_default_password_row_is_owner_only(self) -> None:
        """None means "not asked" — a non-owner's snapshot must not confirm which
        credential this host runs on."""
        assert "db password" not in await self._lines(default_password=None)
        assert "⚠️ db password" in await self._lines(default_password=True)
        assert "✅ db password" in await self._lines(default_password=False)

    async def test_the_default_password_warning_carries_no_detail(self) -> None:
        """The wording IS the mitigation, so the whole row is asserted rather than
        its presence. DEFAULT_POSTGRES_PASSWORD is a literal in this public repo and
        this card is posted in the channel the operator typed in, so spelling the
        warning out publishes the credential itself; -ping carries the full sentence
        for the operator alone."""
        lines = await self._lines(default_password=True)
        row = next(line for line in lines.splitlines() if "db password" in line)
        assert row == "⚠️ db password   see -ping"


class TestRuntimeSampler:
    async def test_starts_from_the_env_default_with_no_toggle(self) -> None:
        """A DEBUG_MODE=true deployment fires no enable transition ever, so a
        transition-only trigger would leave its footers without runtime metrics."""
        sampler = debug.RuntimeSampler()
        try:
            sampler.apply(wanted=True)
            assert sampler.running
        finally:
            await sampler.aclose()

    async def test_both_toggle_transitions(self) -> None:
        sampler = debug.RuntimeSampler()
        try:
            sampler.apply(wanted=True)
            assert sampler.running
            sampler.apply(wanted=False)
            assert not sampler.running
            sampler.apply(wanted=True)
            assert sampler.running
        finally:
            await sampler.aclose()

    async def test_start_is_idempotent(self) -> None:
        sampler = debug.RuntimeSampler()
        try:
            sampler.start()
            first = sampler._task
            sampler.start()
            assert sampler._task is first
        finally:
            await sampler.aclose()

    async def test_aclose_stops_it(self) -> None:
        """Cog teardown must stop it unconditionally, or a reload leaves it
        dripping /proc reads for the life of the process."""
        sampler = debug.RuntimeSampler()
        sampler.start()
        task = sampler._task
        await sampler.aclose()
        assert not sampler.running
        assert task is not None and task.cancelled()

    async def test_snapshot_is_none_before_the_first_tick(self) -> None:
        sampler = debug.RuntimeSampler()
        try:
            sampler.start()
            assert sampler.snapshot is None
        finally:
            await sampler.aclose()

    async def test_tick_produces_a_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug.RuntimeSampler, "INTERVAL_SECS", 0.01)
        sampler = debug.RuntimeSampler()
        try:
            sampler.start()
            for _ in range(200):
                await asyncio.sleep(0.01)
                if sampler.snapshot is not None:
                    break
            assert sampler.snapshot is not None
            assert sampler.snapshot.tasks > 0
            assert sampler.snapshot.pool_workers == 4
        finally:
            await sampler.aclose()

    async def test_a_failing_sample_does_not_end_the_loop(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug.RuntimeSampler, "INTERVAL_SECS", 0.01)
        sampler = debug.RuntimeSampler()
        monkeypatch.setattr(
            sampler, "_sample", MagicMock(side_effect=RuntimeError("boom"))
        )
        try:
            sampler.start()
            await asyncio.sleep(0.05)
            assert sampler.running
        finally:
            await sampler.aclose()

    def test_the_interval_tracks_the_now_playing_tick(self) -> None:
        """Sampling slower than the NP tick re-pushes footers whose numbers have
        not moved."""
        assert debug.RuntimeSampler.INTERVAL_SECS <= (
            config.NOW_PLAYING_UPDATE_INTERVAL_SECS
        )
        assert debug.RuntimeSampler.INTERVAL_SECS >= 1.0  # /proc-read floor
        assert debug.RuntimeSampler.INTERVAL_SECS <= 5.0  # command replies stay fresh

    async def test_the_first_sample_does_not_wait_a_whole_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A full interval of dead air means `-debug --enable` answers with a footer
        carrying no runtime numbers."""
        monkeypatch.setattr(debug.RuntimeSampler, "INTERVAL_SECS", 30.0)
        monkeypatch.setattr(debug, "_FIRST_SAMPLE_SECS", 0.01)
        sampler = debug.RuntimeSampler()
        try:
            sampler.start()
            for _ in range(200):
                await asyncio.sleep(0.01)
                if sampler.snapshot is not None:
                    break
            # Would still be None here if the loop slept INTERVAL_SECS first.
            assert sampler.snapshot is not None
        finally:
            await sampler.aclose()

    async def test_the_first_delay_never_exceeds_the_interval(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A deployment or test that shortens INTERVAL_SECS must not be slowed down
        by the first delay."""
        monkeypatch.setattr(debug.RuntimeSampler, "INTERVAL_SECS", 0.01)
        monkeypatch.setattr(debug, "_FIRST_SAMPLE_SECS", 30.0)
        sampler = debug.RuntimeSampler()
        try:
            sampler.start()
            for _ in range(200):
                await asyncio.sleep(0.01)
                if sampler.snapshot is not None:
                    break
            assert sampler.snapshot is not None
        finally:
            await sampler.aclose()

    async def test_stop_clears_the_cached_snapshot(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Otherwise a re-enable would print metrics sampled before the gap."""
        sampler = debug.RuntimeSampler()
        sampler.start()
        object.__setattr__(sampler, "_snapshot", MagicMock())
        sampler.stop()
        assert sampler.snapshot is None
        await sampler.aclose()


class TestFooterRuntimeSegment:
    def _snapshot(self, **overrides: object) -> debug.RuntimeSnapshot:
        values: dict[str, object] = {
            "cpu_percent": 12.4,
            "mem_percent": 88.2,
            "lag_ms": 2.14,
            "tasks": 87,
            "pool_workers": 4,
        }
        values.update(overrides)
        return debug.RuntimeSnapshot(**values)  # pyright: ignore[reportArgumentType]

    def test_full_segment(self) -> None:
        footer = debug.debug_footer(
            elapsed_ms=412, shard_id=0, runtime=self._snapshot()
        )
        assert footer == (
            "🐞 412 ms · shard 0 · cpu 12% · mem 88% · lag 2.1 ms · tasks 87 · pool 4"
        )

    def test_missing_rates_are_omitted_not_zeroed(self) -> None:
        """Before the sampler's second tick there is no rate to report, and 0%
        would be a lie an operator acts on."""
        footer = debug.debug_footer(
            runtime=self._snapshot(cpu_percent=None, mem_percent=None)
        )
        assert "cpu" not in footer
        assert "mem" not in footer
        assert "lag 2.1 ms" in footer

    def test_no_snapshot_yields_no_runtime_segment(self) -> None:
        assert debug.debug_footer(elapsed_ms=1, shard_id=0) == "🐞 1 ms · shard 0"


class TestContainerMetrics:
    """The Postgres container's CPU/memory, read from the metrics plane.

    Metric names and the label are the OTel docker_stats receiver's as Prometheus
    renames them — verified against the running stack, and NOT cAdvisor's
    `container_cpu_usage_seconds_total{name=...}`, which is what a reader who
    knows the cAdvisor convention would expect.
    """

    @staticmethod
    def _payload(*series: tuple[str, str]) -> dict[str, Any]:
        return {
            "status": "success",
            "data": {
                "resultType": "vector",
                "result": [
                    {
                        "metric": {
                            "__name__": name,
                            "container_name": "discord-postgres",
                        },
                        "value": [1785881580.874, value],
                    }
                    for name, value in series
                ],
            },
        }

    @staticmethod
    def _session(
        payload: object,
        *,
        raises: Optional[Exception] = None,
        body: Optional[bytes] = None,
    ) -> MagicMock:
        """The process-wide session double.

        Serves BYTES, not .json(): the reader takes a capped body off the stream, so
        that a 2s timeout cannot be turned into an unbounded buffer by whatever is
        listening on the configured port. `read(n)` HONOURS n, like the real stream
        does — a double that ignored it would make the cap unobservable.
        """
        response = MagicMock()
        response.raise_for_status = MagicMock(side_effect=raises)
        if body is None:
            body = b"" if payload is None else orjson.dumps(payload)
        served: bytes = body
        response.content = MagicMock()
        response.content.read = AsyncMock(side_effect=lambda n: served[:n])
        get_cm = MagicMock()
        get_cm.__aenter__ = AsyncMock(return_value=response)
        get_cm.__aexit__ = AsyncMock(return_value=False)
        session = MagicMock()
        session.get = MagicMock(return_value=get_cm)
        return session

    def _patch_session(
        self, monkeypatch: pytest.MonkeyPatch, session: MagicMock
    ) -> MagicMock:
        factory = MagicMock(return_value=session)
        monkeypatch.setattr(debug, "_prometheus_session", factory)
        return factory

    async def test_reads_all_three_series_in_one_request(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = self._payload(
            ("container_cpu_utilization_ratio", "3.14"),
            ("container_memory_usage_total_bytes", "100663296"),
            ("container_memory_usage_limit_bytes", "536870912"),
        )
        session = self._session(payload)
        self._patch_session(monkeypatch, session)
        metrics = await debug.read_container_metrics(
            "http://localhost:9090", "discord-postgres"
        )
        assert metrics is not None
        assert metrics.cpu_percent == 3.14
        assert metrics.used_bytes == 100663296
        assert metrics.limit_bytes == 536870912

    async def test_query_selects_the_pinned_container_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Without the label the query matches EVERY container. The name is
        pinned in docker-compose.yml and nothing joins the two — a rename there
        silently empties this row."""
        session = self._session(self._payload())
        self._patch_session(monkeypatch, session)
        await debug.read_container_metrics("http://localhost:9090/", "discord-postgres")
        call = session.get.call_args
        assert call.args[0] == "http://localhost:9090/api/v1/query"
        query = call.kwargs["params"]["query"]
        assert 'container_name="discord-postgres"' in query
        for metric in debug._CONTAINER_METRICS:
            assert metric in query

    async def test_unset_knob_is_the_feature_being_off(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """No knob means no request AT ALL. Asserting only the return value cannot
        tell "the feature is off" from "the feature crashed" — without the guard,
        None.rstrip() raises inside the broad except and this still answers None,
        while every -debug on a deployment with no Prometheus logs a warning.
        """
        factory = self._patch_session(monkeypatch, self._session(self._payload()))
        assert await debug.read_container_metrics(None, "discord-postgres") is None
        assert await debug.read_container_metrics("", "discord-postgres") is None
        factory.assert_not_called()

    async def test_absent_series_is_not_an_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An archive-disabled deployment runs no postgres container at all, so
        an empty result is the ordinary answer."""
        self._patch_session(monkeypatch, self._session(self._payload()))
        assert (
            await debug.read_container_metrics(
                "http://localhost:9090", "discord-postgres"
            )
            is None
        )

    async def test_a_timeout_degrades(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._patch_session(
            monkeypatch, self._session(None, raises=TimeoutError("slow"))
        )
        assert (
            await debug.read_container_metrics(
                "http://localhost:9090", "discord-postgres"
            )
            is None
        )

    async def test_malformed_series_are_skipped_not_fatal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        payload = self._payload(
            ("container_cpu_utilization_ratio", "not-a-number"),
            ("container_memory_usage_total_bytes", "1048576"),
        )
        self._patch_session(monkeypatch, self._session(payload))
        metrics = await debug.read_container_metrics(
            "http://localhost:9090", "discord-postgres"
        )
        assert metrics is not None
        assert metrics.cpu_percent is None
        assert metrics.used_bytes == 1048576

    async def test_the_body_is_taken_off_the_stream_with_a_cap(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`.json()` would buffer whatever arrives inside the 2s timeout. The read
        has to ask for a bounded number of bytes, and one more than the cap so the
        overflow is detectable at all."""
        session = self._session(self._payload(("container_cpu_utilization_ratio", "1")))
        self._patch_session(monkeypatch, session)
        await debug.read_container_metrics("http://localhost:9090", "discord-postgres")
        read = session.get.return_value.__aenter__.return_value.content.read
        read.assert_awaited_once_with(debug._PROMETHEUS_MAX_BYTES + 1)

    async def test_an_oversized_body_is_refused_rather_than_parsed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ClientTimeout bounds TIME, not BYTES, and the default target is a loopback
        port on a host-networked bot — anything that can bind :9090 could feed this
        gigabytes.

        The REASON is asserted, not the return value: a truncated body fails to parse
        and answers None either way, so dropping the cap check is invisible from the
        outside. This is the one observable that separates them.
        """
        self._patch_session(
            monkeypatch,
            self._session(None, body=b"x" * (debug._PROMETHEUS_MAX_BYTES + 1)),
        )
        with patch.object(debug, "log") as log:
            assert (
                await debug.read_container_metrics(
                    "http://localhost:9090", "discord-postgres"
                )
                is None
            )
        assert "exceeded" in log.warning.call_args.args[0]

    def test_rendered_lines_put_cpu_before_mem(self) -> None:
        lines = debug._container_lines(
            debug.ContainerMetrics(
                cpu_percent=3.14, used_bytes=100663296, limit_bytes=536870912
            )
        )
        assert lines[0] == "cpu          3.1% — prometheus"
        assert lines[1] == "mem          96 MB / 512 MB (19%) — prometheus"

    def test_no_source_says_so_rather_than_showing_zero(self) -> None:
        """Zero would be a number an operator acts on; the absence must be named."""
        lines = debug._container_lines(None)
        assert all("n/a (no metrics source)" in line for line in lines)


class TestPrometheusSession:
    """The session factory itself, which nothing executed: every TestContainerMetrics
    case patches `_prometheus_session` away, so the timeout, the cache and the close
    all survived deletion."""

    @pytest.fixture(autouse=True)
    async def _isolate_the_session_cache(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> AsyncIterator[None]:
        """Start from an empty process-wide cache and never leave a real
        ClientSession unclosed — aiohttp warns on collection, and this suite runs
        with filterwarnings = error."""
        monkeypatch.setattr(debug, "_prometheus_session_cache", None)
        yield
        await debug.close_prometheus_session()

    async def test_the_session_carries_the_query_timeout(self) -> None:
        """aiohttp's default total timeout is FIVE MINUTES. Without this kwarg a
        black-holed metrics endpoint holds the Postgres probe far past -debug's own
        8s deadline, and the compose comment calls that port unauthenticated."""
        session = debug._prometheus_session()
        assert session.timeout.total == debug._PROMETHEUS_TIMEOUT_SECS

    async def test_one_session_serves_the_process(self) -> None:
        """A session per query paid an extra connect every time — a full TCP (plus
        TLS) handshake against anything not on loopback."""
        first = debug._prometheus_session()
        assert debug._prometheus_session() is first

    async def test_a_closed_session_is_replaced(self) -> None:
        """cog_unload closes it; the next -debug must not be handed the corpse."""
        first = debug._prometheus_session()
        await first.close()
        second = debug._prometheus_session()
        assert second is not first and not second.closed

    async def test_close_really_closes_and_clears_the_cache(self) -> None:
        """The one test that drove cog_unload had a None cache, so a no-op body was
        green while leaking a connector per reload."""
        session = debug._prometheus_session()
        await debug.close_prometheus_session()
        assert session.closed
        assert debug._prometheus_session_cache is None

    async def test_closing_twice_is_safe(self) -> None:
        """cog_unload can run after a teardown that already closed it."""
        debug._prometheus_session()
        await debug.close_prometheus_session()
        await debug.close_prometheus_session()
        assert debug._prometheus_session_cache is None


class TestRateHumanizers:
    """The formatting helpers behind the native rows, directly: the renderer
    tests exercise only a handful of values, and the plan waived a temp-spill
    integration test on the promise that these are unit-covered."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (0.44, "0.4"),
            (9.94, "9.9"),
            (9.99, "10"),  # decided off the rounded value — never "10.0"
            (41.0, "41"),
            (999.99, "1000"),
        ],
    )
    def test_rate(self, value: float, expected: str) -> None:
        assert debug._rate(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0"),
            (500, "500"),
            (999.94, "1000"),
            (999.99, "1.0k"),  # rounded-mantissa promotion at the unit seam
            (1_000, "1.0k"),
            (12_400, "12.4k"),
            (999_940, "999.9k"),
            (999_960, "1.0M"),  # never "1000.0k"
            (2_500_000, "2.5M"),
        ],
    )
    def test_count_rate(self, value: float, expected: str) -> None:
        assert debug._count_rate(value) == expected

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (0, "0 B/s"),
            (512, "512 B/s"),
            (1023.4, "1023 B/s"),
            (1023.6, "1 KB/s"),  # rounds into the next unit — never "1024 B/s"
            (2048, "2 KB/s"),
            (1_048_575, "1.0 MB/s"),  # never "1024 KB/s"
            (5_242_880, "5.0 MB/s"),
        ],
    )
    def test_bytes_rate(self, value: float, expected: str) -> None:
        assert debug._bytes_rate(value) == expected


class TestPostgresBlockWithContainerMetrics:
    def test_container_rows_lead_the_block(self) -> None:
        container = debug.ContainerMetrics(
            cpu_percent=3.1, used_bytes=1048576, limit_bytes=2097152
        )
        lines = debug.postgres_lines(container, None, _pg_stats())
        assert lines[0].startswith("cpu")
        assert lines[1].startswith("mem")
        assert lines[2].startswith("load")

    def test_sql_still_answers_when_the_metrics_stack_is_down(self) -> None:
        """Why the block is hybrid rather than all-PromQL: an LGTM outage costs the
        cpu/mem pair, not the whole block — and the native rows are exactly the
        signal that stays live with the `metrics` profile off."""
        before = _pg_stats()
        after = _pg_stats(xacts_total=1_082, monotonic=102.0)
        lines = "\n".join(debug.postgres_lines(None, before, after))
        assert "n/a (no metrics source)" in lines
        assert "db 128 MB" in lines
        assert "~412083 plays" in lines
        assert "41 tx/s" in lines


class TestPostgresProbeOrchestration:
    """The two-sample bracket in _postgres_probe, mirroring the Redis probe's
    shape: first sample → window sleep → second sample gathered with Prometheus,
    all inside one outer timeout."""

    def _inputs(self, archive: Any, **overrides: Any) -> DebugInputs:
        kwargs: dict[str, Any] = dict(
            debug_enabled=False,
            debug_overridden=False,
            players=0,
            archive=archive,
            archive_enabled=True,
        )
        kwargs.update(overrides)
        return DebugInputs(**kwargs)

    async def test_rates_render_from_both_samples(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug, "_PG_WINDOW_SECS", 0.0)
        archive = MagicMock()
        archive.stats = AsyncMock(
            side_effect=[
                _pg_stats(),
                _pg_stats(xacts_total=1_082, monotonic=102.0),
            ]
        )
        lines = "\n".join(await debug._postgres_probe(self._inputs(archive)))
        assert archive.stats.await_count == 2
        assert "41 tx/s" in lines

    async def test_first_sample_failure_costs_only_the_rates(self) -> None:
        """Asymmetric on purpose: `before` failing must not take down the instant
        fields, which need only the second sample. Runs with the REAL window and a
        1s timeout to prove the probe skips the sleep — nothing to rate over means
        nothing to wait for."""
        archive = MagicMock()
        archive.stats = AsyncMock(side_effect=[RuntimeError("boom"), _pg_stats()])
        async with asyncio.timeout(1):
            lines = "\n".join(await debug._postgres_probe(self._inputs(archive)))
        assert "busy unknown" in lines
        assert "throughput   unknown" in lines
        assert "db 128 MB" in lines
        assert "unavailable" not in lines

    async def test_second_sample_failure_degrades_the_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other half of the asymmetry: `after` carries the instant fields, so
        its failure leaves nothing to render — the block degrades whole."""
        monkeypatch.setattr(debug, "_PG_WINDOW_SECS", 0.0)
        archive = MagicMock()
        archive.stats = AsyncMock(side_effect=[_pg_stats(), RuntimeError("boom")])
        blocks = await debug._postgres_blocks(self._inputs(archive))
        assert blocks["Postgres"] == ["unavailable (RuntimeError)"]

    async def test_the_probe_timeout_degrades_the_block(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The outer bound exists to release the archive's read slots: a hung
        Postgres must cost this block, not ~32s of held read capacity."""
        monkeypatch.setattr(debug, "_PG_PROBE_TIMEOUT_SECS", 0.05)

        async def never() -> Any:
            await asyncio.Event().wait()

        archive = MagicMock()
        archive.stats = never
        blocks = await debug._postgres_blocks(self._inputs(archive))
        assert blocks["Postgres"] == ["unavailable (TimeoutError)"]

    async def test_prometheus_rides_the_second_sample(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(debug, "_PG_WINDOW_SECS", 0.0)
        fetch = AsyncMock(return_value=None)
        monkeypatch.setattr(debug, "read_container_metrics", fetch)
        archive = MagicMock()
        archive.stats = AsyncMock(side_effect=[_pg_stats(), _pg_stats()])
        await debug._postgres_probe(
            self._inputs(archive, prometheus_url="http://localhost:9090")
        )
        fetch.assert_awaited_once_with(
            "http://localhost:9090", debug._POSTGRES_CONTAINER
        )

    async def test_disabled_archive_reads_as_a_choice(self) -> None:
        """Mirrors -ping's OFF row: the ship default made this choice, and it must
        not read as a fault."""
        blocks = await debug._postgres_blocks(self._inputs(None, archive_enabled=False))
        assert blocks["Postgres"] == ["off (archive disabled)"]

    async def test_enabled_but_absent_archive_is_na(self) -> None:
        blocks = await debug._postgres_blocks(self._inputs(None))
        assert blocks["Postgres"] == ["n/a"]


class TestGuardsMutationTestingFound:
    """Small guards whose absence a green suite could not distinguish.

    Each of these survived a mutation run: the behaviour was correct, but nothing
    asserted it, so the next edit to that line would have shipped silently.
    """

    async def test_an_async_collector_degrades_like_a_sync_one(self) -> None:
        """_safe_block guards the sync collectors and had a test; the ASYNC wrapper
        did not. Without it a raise inside checks_lines escapes to the probe, which
        fails the probe's ENTIRE block set — so one broken collector would take
        Redis down with Checks instead of costing only its own field."""

        async def boom() -> list[str]:
            raise RuntimeError("collector broke")

        assert await debug._safe_block_async("checks", boom) == [
            "unavailable (RuntimeError)"
        ]

    async def test_one_broken_collector_leaves_its_sibling_block_intact(
        self, mock_ctx: MagicMock
    ) -> None:
        """The consequence, end to end: Checks and Redis share a probe."""
        mock_ctx.guild.voice_client = None
        with patch.object(debug, "checks_lines", side_effect=RuntimeError("boom")):
            embed = await _snapshot(
                mock_ctx,
                DebugInputs(
                    debug_enabled=False,
                    debug_overridden=False,
                    players=0,
                    operator=True,
                ),
            )
        rendered = {f.name: (f.value or "") for f in embed.fields}
        assert "unavailable (RuntimeError)" in rendered["Checks"]
        # Redis still renders its OWN degradation (there is no Redis in this
        # fixture) rather than inheriting the sibling collector's failure.
        assert "RuntimeError" not in rendered["Redis"]

    async def test_the_first_sample_reports_no_cpu_rather_than_zero(self) -> None:
        """A rate needs two samples. Reporting 0.0% on the first tick tells an
        operator "idle" at the exact moment they are investigating load."""
        sampler = debug.RuntimeSampler()
        snapshot = sampler._sample(1.5)
        assert snapshot.cpu_percent is None

    async def test_a_guild_that_has_played_nothing_is_not_an_alarm(self) -> None:
        """TTL -2 is "no such key" — an ordinary fresh guild, not a violation of the
        PERSIST invariant. Rendering it as a warning cries wolf on the most common
        state of a new deployment, and a check row that cries wolf gets ignored."""
        store = MagicMock()
        store.history_ttl = AsyncMock(return_value=-2)
        lines = await debug.checks_lines(
            None,
            redis=None,
            store=store,
            archive_enabled=False,
            default_password=None,
        )
        row = next(line for line in lines if "history ttl" in line)
        assert "no history yet" in row
        assert "⚠️" not in row

    def test_a_username_only_url_is_still_redacted(self) -> None:
        """`redis://admin@host` carries no password, but the username is still a
        credential half and renders in an owner-visible block."""
        assert "admin" not in debug.redact_url("redis://admin@host:6379")

    async def test_an_rdb_only_redis_does_not_fail_the_persistence_check(self) -> None:
        """A Redis with no AOF omits aof_last_write_status entirely. Treating the
        absent field as failure pins this row to a permanent warning."""
        sample = debug.RedisSample(
            info={"maxmemory_policy": "volatile-lru", "rdb_last_bgsave_status": "ok"},
            monotonic=0.0,
        )
        lines = await debug.checks_lines(
            sample,
            redis=None,
            store=None,
            archive_enabled=False,
            default_password=None,
        )
        row = next(line for line in lines if "persistence" in line)
        assert "⚠️" not in row

    def test_a_footer_with_nothing_to_say_is_empty(self) -> None:
        """Not a bare mark. A DM reply in the first seconds after DEBUG_MODE=true has
        no elapsed time, no shard and no snapshot — the marker alone is noise."""
        assert debug.debug_footer() == ""

    def test_a_warming_up_voice_client_does_not_claim_an_average(self) -> None:
        """latency is inf until the first heartbeat ACK. If average_latency happens
        to answer anyway, "warming up (avg 3 ms)" is a contradiction."""
        vc = MagicMock(spec=discord.VoiceClient)
        vc.latency = float("inf")
        vc.average_latency = 0.003
        assert debug._voice_latency_text(vc) == "warming up"

    def test_the_shard_list_is_capped_with_a_marker(self) -> None:
        """An unbounded list would push the block past Discord's field cap on a
        many-shard bot; the marker is what says the list was cut."""
        bot = MagicMock()
        bot.latencies = [(i, 0.05) for i in range(9)]
        line = next(
            line for line in debug.discord_lines(bot, players=0) if "gateway" in line
        )
        assert line.count("#") == 4
        assert "…" in line

    async def test_the_password_row_is_silent_without_a_database(self) -> None:
        """A ✅ "db password not the default" on a deployment that runs no Postgres
        at all is an answer to a question nobody asked."""
        lines = await debug.checks_lines(
            None,
            redis=None,
            store=None,
            archive_enabled=False,
            default_password=False,
        )
        assert not any("db password" in line for line in lines)

    def test_a_second_decoration_does_not_stack_two_trace_ids(self) -> None:
        """_command_error writes "trace: <id>"; the footer writes "trace <id>". Only
        the first spelling had a test, so the second arm could be dropped silently."""
        embed = discord.Embed(title="x")
        trace_id = 0x4BF92F3577B34DA6A3CE929D0E0E4736
        hex_id = "4bf92f3577b34da6a3ce929d0e0e4736"
        span = trace_api.NonRecordingSpan(
            trace_api.SpanContext(
                trace_id=trace_id,
                span_id=0x00F067AA0BA902B7,
                is_remote=False,
                trace_flags=trace_api.TraceFlags(trace_api.TraceFlags.SAMPLED),
            )
        )
        embed.set_footer(text=f"trace {hex_id}")
        debug.decorate_embeds([embed], span=span, elapsed_ms=1.0, shard_id=None)
        assert (embed.footer.text or "").count(hex_id) == 1

    def test_each_skeleton_gets_its_own_placeholder_list(self) -> None:
        """Shared mutable default: two concurrent -debug invocations would otherwise
        write into one another's blocks."""
        first = list(debug._PENDING_LINES)
        first.append("mutated")
        assert debug._PENDING_LINES == ["⏳ collecting…"]

    async def test_the_outbox_row_says_so_when_there_is_no_redis(self) -> None:
        """Distinct from a Redis that answered zero — the broad except below would
        otherwise swallow the AttributeError and report the same thing."""
        row = await debug._outbox_check(None, archive_enabled=True)
        assert "no Redis" in row


class TestDebugSamplerLifecycle:
    """The sampler feeds cpu/mem/lag into every decorated footer. It must run
    exactly while some guild wants it — no sooner, and not after unload."""

    async def test_enabling_starts_the_sampler(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        assert music_bot.debug_settings._sampler.running is False
        await command_callback(MusicBotCog.debug)(music_bot, mock_ctx, arg="--enable")
        assert music_bot.debug_settings._sampler.running is True
        await music_bot.debug_settings._sampler.aclose()

    async def test_the_last_disable_stops_it(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        await command_callback(MusicBotCog.debug)(music_bot, mock_ctx, arg="--enable")
        await command_callback(MusicBotCog.debug)(music_bot, mock_ctx, arg="--disable")
        assert music_bot.debug_settings._sampler.running is False

    async def test_another_guild_still_wanting_it_keeps_it_running(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        music_bot.debug_settings._overrides[999] = True
        music_bot.debug_settings.sync_sampler()
        await command_callback(MusicBotCog.debug)(music_bot, mock_ctx, arg="--disable")
        assert music_bot.debug_settings._sampler.running is True
        await music_bot.debug_settings._sampler.aclose()

    async def test_the_env_default_starts_it_at_cog_load(
        self, music_bot: MusicBotCog
    ) -> None:
        """DEBUG_MODE=true fires no enable transition, ever — so cog_load has to
        evaluate the run condition too, not only the toggle path."""
        music_bot.debug_settings._default = True
        await music_bot.cog_load()
        assert music_bot.debug_settings._sampler.running is True
        await music_bot.debug_settings._sampler.aclose()

    async def test_cog_unload_stops_it(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """Without this a cog reload leaves a sampler dripping /proc reads forever.

        `cancelled()`, never `cancelled() or done()`: the latter is just `done()`,
        which a task that simply RETURNED also satisfies. `running` is no better —
        stop() nulls the reference, so it reads False whether or not the loop ever
        stopped. Only the task object still held here can tell them apart."""
        await command_callback(MusicBotCog.debug)(music_bot, mock_ctx, arg="--enable")
        sampler = music_bot.debug_settings._sampler
        task = sampler._task
        assert task is not None
        # Seed a snapshot so teardown clearing it is observable: a dead sampler
        # must not keep feeding footers a sample nothing will ever refresh.
        sampler._snapshot = sampler._sample(1.5)
        await music_bot.cog_unload()
        assert task.cancelled()
        assert sampler._task is None
        assert sampler.snapshot is None

    async def test_unload_cancels_an_in_flight_hydration(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """_hydrate_debug ends in sync_sampler, and nothing else
        ever cancels _restore_tasks. Parked mid-read across cog_unload it resumed
        afterwards and started a FRESH sampler task holding the dead cog — a
        permanent leak per reload_extension."""
        cog = music_bot_with_redis
        cast(Any, cog.bot).guilds = [MagicMock(spec=discord.Guild, id=42)]
        cog.debug_settings._overrides = {42: True}
        cog.debug_settings.sync_sampler()
        assert cog.debug_settings._sampler.running is True
        reading, released = asyncio.Event(), asyncio.Event()

        async def parks_until_released(*_a: object, **_k: object) -> dict[int, Any]:
            reading.set()
            await released.wait()
            return {42: GuildConfig(debug_mode=True)}

        with patch("src.debug.read_guild_configs", new=parks_until_released):
            task = spawn_background(cog._hydrate_debug(), cog._restore_tasks)
            await reading.wait()
            await cog.cog_unload()
            # Would resume the hydration if it had survived the unload.
            released.set()
            await asyncio.sleep(0)

        assert task.cancelled()
        assert cog.debug_settings._sampler.running is False
        assert not cog._restore_tasks

    async def test_background_tasks_are_cancelled_before_the_sampler_closes(
        self, music_bot: MusicBotCog
    ) -> None:
        """The order is the fix, not just the cancel: aclose() awaits, so a task
        still runnable when it starts resumes INSIDE it and can start a fresh
        sampler that the later cancel no longer reaches."""
        order: list[str] = []
        started = asyncio.Event()

        async def sentinel() -> None:
            started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                order.append("cancelled")
                raise

        sampler = music_bot.debug_settings._sampler
        real_aclose = sampler.aclose

        async def recording_aclose() -> None:
            order.append("aclose")
            await real_aclose()

        spawn_background(sentinel(), music_bot._restore_tasks)
        await started.wait()
        with patch.object(sampler, "aclose", recording_aclose):
            await music_bot.cog_unload()

        assert order == ["cancelled", "aclose"]

    async def test_the_sampled_snapshot_reaches_the_footer(
        self, music_bot: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """The wiring, end to end: sampler → cog.debug_settings.snapshot → debug_footer."""
        await command_callback(MusicBotCog.debug)(music_bot, mock_ctx, arg="--enable")
        try:
            sampler = music_bot.debug_settings._sampler
            sampler._snapshot = sampler._sample(1.5)
            snapshot = music_bot.debug_settings.snapshot
            assert snapshot is not None
            footer = debug.debug_footer(
                span=None, elapsed_ms=12.0, shard_id=0, runtime=snapshot
            )
            assert "cpu " in footer or "mem " in footer
            assert "tasks " in footer
        finally:
            await music_bot.debug_settings._sampler.aclose()


class TestDebugModeIsPerGuildAndDurable:
    """DEBUG_MODE is the default every guild starts from; each guild can pin its own
    choice, and that choice now outlives a restart."""

    @staticmethod
    def _guilds(cog: MusicBotCog, *ids: int) -> None:
        # bot is a MagicMock here; Bot.guilds is a read-only property on the
        # real class, which the checker sees through the annotation.
        cast(Any, cog.bot).guilds = [MagicMock(spec=discord.Guild, id=i) for i in ids]

    async def test_the_env_default_turns_every_guild_on(
        self, music_bot: MusicBotCog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(music_bot.debug_settings, "_default", True)
        monkeypatch.setattr(music_bot.debug_settings, "_overrides", {})
        assert music_bot.debug_settings.enabled(123) is True
        assert music_bot.debug_settings.enabled(456) is True

    async def test_without_the_env_default_a_guild_must_opt_in(
        self, music_bot: MusicBotCog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(music_bot.debug_settings, "_default", False)
        monkeypatch.setattr(music_bot.debug_settings, "_overrides", {123: True})
        assert music_bot.debug_settings.enabled(123) is True
        assert music_bot.debug_settings.enabled(456) is False

    async def test_a_guild_can_opt_out_while_the_host_default_is_on(
        self, music_bot: MusicBotCog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The reason the stored value is tri-state. A plain bool cannot tell "never
        chose" from "chose off", so this guild would be dragged back on."""
        monkeypatch.setattr(music_bot.debug_settings, "_default", True)
        monkeypatch.setattr(music_bot.debug_settings, "_overrides", {123: False})
        assert music_bot.debug_settings.enabled(123) is False
        assert music_bot.debug_settings.enabled(456) is True

    async def test_the_toggle_writes_through_to_redis(
        self, music_bot_with_redis: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild.id = 42
        await music_bot_with_redis._toggle_debug_mode(
            mock_ctx, debug.DebugAction.ENABLE
        )
        store = GuildRedisStore(cast(Any, music_bot_with_redis.redis), 42)
        assert (await store.get_config()).debug_mode is True

    async def test_a_stored_choice_is_restored_on_startup(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """The point of the whole change: the setting used to die with the process."""
        redis = cast(Any, music_bot_with_redis.redis)
        await GuildRedisStore(redis, 111).set_debug_mode(True)
        await GuildRedisStore(redis, 222).set_debug_mode(False)
        self._guilds(music_bot_with_redis, 111, 222, 333)
        music_bot_with_redis.debug_settings._overrides = {}

        await music_bot_with_redis._hydrate_debug()

        assert music_bot_with_redis.debug_settings._overrides == {111: True, 222: False}
        # 333 never chose, so it is absent rather than cached — caching it would
        # freeze it against a later DEBUG_MODE change.
        assert 333 not in music_bot_with_redis.debug_settings._overrides

    async def test_a_guild_with_no_stored_choice_still_follows_a_changed_default(
        self, music_bot_with_redis: MusicBotCog, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._guilds(music_bot_with_redis, 333)
        await music_bot_with_redis._hydrate_debug()
        monkeypatch.setattr(music_bot_with_redis.debug_settings, "_default", True)
        assert music_bot_with_redis.debug_settings.enabled(333) is True

    async def test_hydration_survives_an_unreachable_redis(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """read_guild_configs reports a failed batch by omission, so startup
        degrades to the host default rather than dying."""
        self._guilds(music_bot_with_redis, 111)
        with patch.object(
            cast(Any, music_bot_with_redis.redis),
            "pipeline",
            side_effect=RuntimeError("down"),
        ):
            await music_bot_with_redis._hydrate_debug()
        assert music_bot_with_redis.debug_settings._overrides == {}

    async def test_a_failed_read_does_not_discard_a_correct_stored_choice(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """The failure this whole read path is shaped around. A guild's config read
        failing is NOT "this guild never chose": collapsing the two deleted a correct
        cached value and reverted the guild to the host default for the rest of the
        process, while Redis still held the opposite. on_ready re-fires on every
        session loss, so a single blink was enough.

        Starts from a POPULATED cache on purpose — asserting `== {}` from an empty
        one cannot see a correct entry being destroyed, which is why the suite was
        green while this was broken.
        """
        redis = cast(Any, music_bot_with_redis.redis)
        await GuildRedisStore(redis, 111).set_debug_mode(False)
        self._guilds(music_bot_with_redis, 111)
        music_bot_with_redis.debug_settings._overrides = {111: False}

        with patch.object(redis, "pipeline", side_effect=RuntimeError("down")):
            await music_bot_with_redis._hydrate_debug()

        assert music_bot_with_redis.debug_settings._overrides == {111: False}
        assert music_bot_with_redis.debug_settings.enabled(111) is False

    async def test_a_read_that_succeeds_still_evicts_a_removed_choice(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """The other half: skipping on failure must not turn into never evicting.
        A choice cleared out from under the process has to stop being cached, or the
        guild is frozen against a later DEBUG_MODE change."""
        self._guilds(music_bot_with_redis, 111)
        music_bot_with_redis.debug_settings._overrides = {111: True}

        await music_bot_with_redis._hydrate_debug()

        assert 111 not in music_bot_with_redis.debug_settings._overrides

    async def test_a_toggle_during_hydration_is_not_overwritten(
        self, music_bot_with_redis: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """Hydration reads Redis and applies the result across an await. A toggle
        landing in that window used to be overwritten by the value that was true
        before the user ran it — they are told "saved for this server", Redis agrees,
        and the footer never appears until the next session loss."""
        redis = cast(Any, music_bot_with_redis.redis)
        await GuildRedisStore(redis, 42).set_debug_mode(False)
        self._guilds(music_bot_with_redis, 42)
        music_bot_with_redis.debug_settings._overrides = {42: False}
        mock_ctx.guild.id = 42

        async def read_then_user_toggles(*_a: object, **_k: object) -> dict[int, Any]:
            # The read has resolved; the toggle lands before the loop applies it.
            await music_bot_with_redis._toggle_debug_mode(
                mock_ctx, debug.DebugAction.ENABLE
            )
            return {42: GuildConfig(debug_mode=False)}  # what the read saw

        with patch("src.debug.read_guild_configs", new=read_then_user_toggles):
            await music_bot_with_redis._hydrate_debug()

        assert music_bot_with_redis.debug_settings.enabled(42) is True

    async def test_a_failed_write_is_reported_not_claimed(
        self, music_bot_with_redis: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """Telling a guild "on" when the write never landed reads as the bot
        ignoring them after the next restart."""
        mock_ctx.guild.id = 42
        with patch.object(
            cast(Any, music_bot_with_redis.redis),
            "pipeline",
            side_effect=RuntimeError("down"),
        ):
            await music_bot_with_redis._toggle_debug_mode(
                mock_ctx, debug.DebugAction.ENABLE
            )
        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert "could not be saved" in description
        # Still applied in memory for this process.
        assert music_bot_with_redis.debug_settings.enabled(42) is True

    async def test_debug_reports_a_failed_write_as_session_only(
        self, music_bot_with_redis: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        """The toggle warns "it could not be saved"; -debug used to render "saved
        here" one message later, because debug_overridden is just cache membership
        and the cache is written whether or not Redis took it. The one command whose
        job is to report reality contradicted the one that just changed it."""
        mock_ctx.guild.id = 42
        with patch.object(
            cast(Any, music_bot_with_redis.redis),
            "pipeline",
            side_effect=RuntimeError("down"),
        ):
            await music_bot_with_redis._toggle_debug_mode(
                mock_ctx, debug.DebugAction.ENABLE
            )

        inputs = await music_bot_with_redis._debug_inputs(mock_ctx)

        assert inputs.debug_overridden is True
        assert inputs.debug_persisted is False
        assert debug.mode_source(True, persisted=False) == "this session only"

    async def test_a_write_that_lands_reports_as_saved(
        self, music_bot_with_redis: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild.id = 42
        await music_bot_with_redis._toggle_debug_mode(
            mock_ctx, debug.DebugAction.ENABLE
        )
        inputs = await music_bot_with_redis._debug_inputs(mock_ctx)
        assert inputs.debug_persisted is True

    async def test_hydration_clears_a_stale_unpersisted_mark(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """Redis came back and the durable copy agrees, so the warning must stop."""
        redis = cast(Any, music_bot_with_redis.redis)
        await GuildRedisStore(redis, 111).set_debug_mode(True)
        self._guilds(music_bot_with_redis, 111)
        music_bot_with_redis.debug_settings._unpersisted.add(111)

        await music_bot_with_redis._hydrate_debug()

        assert 111 not in music_bot_with_redis.debug_settings._unpersisted

    async def test_a_successful_write_says_it_is_saved(
        self, music_bot_with_redis: MusicBotCog, mock_ctx: MagicMock
    ) -> None:
        mock_ctx.guild.id = 42
        await music_bot_with_redis._toggle_debug_mode(
            mock_ctx, debug.DebugAction.ENABLE
        )
        description = mock_ctx.send.await_args.kwargs["embed"].description
        assert "saved for this server" in description
        assert "could not be saved" not in description

    async def test_leaving_a_guild_drops_the_stored_choice(
        self, music_bot_with_redis: MusicBotCog
    ) -> None:
        """Config carries no TTL, so nothing else would ever remove it — and a
        rejoin would silently resume a setting nobody there chose."""
        redis = cast(Any, music_bot_with_redis.redis)
        await GuildRedisStore(redis, 111).set_debug_mode(True)
        music_bot_with_redis.debug_settings._overrides = {111: True}
        guild = MagicMock(spec=discord.Guild, id=111)

        await music_bot_with_redis.on_guild_remove(guild)

        assert 111 not in music_bot_with_redis.debug_settings._overrides
        assert (await GuildRedisStore(redis, 111).get_config()).debug_mode is None
