"""Debug mode's footer: what every reply grows while the mode is on.

Observation-only by rule, so what is asserted here is entirely what is DISPLAYED.
The seam that applies it to real command responses is tested in test_context.py.
"""

import asyncio
import os
import re
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional, cast
from unittest.mock import AsyncMock, MagicMock, patch

import discord
import pytest
from redis.asyncio import Redis
from discord.ext import commands
from opentelemetry import trace as trace_api

from src import debug
from src.history_archive import ArchiveStats
from src.musicplayer import MusicPlayer
from src.debug import (
    DebugInputs,
    _CONFIG_ALLOWLIST,
    _ConfigKind,
    _codeblock_fields,
    config_lines,
    discord_lines,
    guild_lines,
    redact_url,
    render_config_value,
    run_debug_dashboard,
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


class TestDebugFooter:
    """Every segment is optional because every one has a genuine absent case: a
    send outside any command has no elapsed time, a DM has no shard, and an
    unsampled span has no trace id."""

    def test_nothing_known_renders_nothing(self) -> None:
        """An empty string, not a lone marker — a footer that says only "🐞" is
        noise on every reply and tells the operator nothing."""
        assert debug.debug_footer() == ""

    def test_elapsed_and_shard_render(self) -> None:
        footer = debug.debug_footer(elapsed_ms=12.4, shard_id=3)
        assert "12 ms" in footer
        assert "shard 3" in footer

    def test_shard_zero_is_not_mistaken_for_absent(self) -> None:
        """Shard 0 is a real shard and falsy; `is not None` is what separates them."""
        assert "shard 0" in debug.debug_footer(shard_id=0)

    def test_an_embeds_own_footer_is_kept(self) -> None:
        embed = discord.Embed(description="x")
        embed.set_footer(text="Requested by someone")
        debug.decorate_embeds([embed], elapsed_ms=5.0)
        text = embed.footer.text or ""
        assert text.startswith("Requested by someone")
        assert "5 ms" in text

    def test_an_icon_url_survives_decoration(self) -> None:
        """set_footer replaces the whole footer object, so the icon has to be
        carried across explicitly or every decorated reply silently loses it."""
        embed = discord.Embed(description="x")
        embed.set_footer(text="by someone", icon_url="https://example.invalid/a.png")
        debug.decorate_embeds([embed], elapsed_ms=5.0)
        assert embed.footer.icon_url == "https://example.invalid/a.png"

    def test_an_embed_with_nothing_to_add_is_left_alone(self) -> None:
        embed = discord.Embed(description="x")
        debug.decorate_embeds([embed])
        assert embed.footer.text is None

    def test_the_footer_is_truncated_to_discords_cap(self) -> None:
        embed = discord.Embed(description="x")
        embed.set_footer(text="A" * (debug.FOOTER_LIMIT + 500))
        debug.decorate_embeds([embed], elapsed_ms=1.0)
        assert len(embed.footer.text or "") <= debug.FOOTER_LIMIT

    def test_every_embed_in_the_list_is_decorated(self) -> None:
        embeds = [discord.Embed(description=str(i)) for i in range(3)]
        debug.decorate_embeds(embeds, elapsed_ms=7.0)
        assert all("7 ms" in (e.footer.text or "") for e in embeds)


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

    def test_an_unsampled_span_contributes_nothing(self) -> None:
        """INVALID_SPAN has an all-zero trace id, which trace_id_of reports as "".
        Rendering it would put a meaningless id in front of a user who is being
        told to paste it to an operator."""
        assert debug.debug_footer(span=trace_api.INVALID_SPAN) == ""


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
        # The neighbouring blocks are untouched — one broken collector costs one row.
        assert [f.name for f in embed.fields].count("Discord") == 1


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


class TestSnapshotEmbed:
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

    async def test_renders_every_block_in_a_guild(self, mock_ctx: MagicMock) -> None:
        """Order is fixed by _BLOCK_ORDER, so a block added later has to choose its
        place rather than landing wherever the dict happened to put it."""
        mock_ctx.guild.voice_client = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(
                debug_enabled=False,
                debug_overridden=False,
                players=1,
                operator=True,
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
        ]

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

    async def test_footer_carries_environment(self, mock_ctx: MagicMock) -> None:
        mock_ctx.guild = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert embed.footer.text is not None
        assert embed.footer.text.startswith("environment: ")


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


class TestOperatorGate:
    """`-debug` is reachable by any user in any guild, and by DM. Everything that
    describes the HOST rather than the caller's own server is owner-only."""

    _HOST_BLOCKS = ("Build", "Config", "Runtime", "Discord", "Redis", "Postgres")

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
        the block and then drops it still reads the host, and still pays the IO."""
        mock_ctx.guild.voice_client = None
        with (
            patch.object(debug, "config_lines") as config,
            patch.object(debug, "runtime_lines") as runtime,
            patch.object(debug, "redis_lines") as redis_block,
            patch.object(debug, "discord_lines") as discord_block,
            patch.object(debug, "build_lines") as build,
        ):
            await _snapshot(
                mock_ctx,
                DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
            )
        for collector in (config, runtime, redis_block, discord_block, build):
            collector.assert_not_called()

    async def test_a_non_operators_snapshot_carries_no_host_detail(
        self, mock_ctx: MagicMock
    ) -> None:
        """The disclosure this gate exists for. The sentinel stands in for every
        host-describing row."""
        mock_ctx.guild.voice_client = None
        with patch.object(debug, "discord_lines", return_value=["shards  sentinel"]):
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
    async def test_every_host_block_is_absent_for_a_non_operator(
        self, mock_ctx: MagicMock, block: str
    ) -> None:
        """Parametrized per block so a block added later has to decide which side of
        this gate it is on rather than defaulting to public."""
        mock_ctx.guild.voice_client = None
        embed = await _snapshot(
            mock_ctx,
            DebugInputs(debug_enabled=False, debug_overridden=False, players=0),
        )
        assert block not in [f.name for f in embed.fields]


class TestTheSnapshotDoesNotWaitForItsIO:
    """-debug sends a skeleton and fills it in. The point is that a sick dependency
    delays one BLOCK, not the whole reply."""

    @staticmethod
    def _operator(**kw: Any) -> DebugInputs:
        return DebugInputs(
            debug_enabled=False, debug_overridden=False, players=0, operator=True, **kw
        )

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
        self, mock_ctx: MagicMock
    ) -> None:
        """The whole contract: the skeleton is out while Build is still shelling
        out to git."""
        mock_ctx.guild.voice_client = None
        gate = asyncio.Event()
        message = MagicMock(spec=discord.Message)
        message.edit = AsyncMock()
        mock_ctx.channel.send = AsyncMock(return_value=message)

        async def slow_build() -> dict[str, list[str]]:
            await gate.wait()
            return {"Build": ["commit  abc123"]}

        with patch.object(debug, "_build_blocks", slow_build):
            task = asyncio.create_task(run_debug_dashboard(mock_ctx, self._operator()))
            async with asyncio.timeout(3):
                while mock_ctx.channel.send.await_count == 0:
                    await asyncio.sleep(0)
                call = mock_ctx.channel.send.await_args
                assert call is not None
                sent = call.kwargs["embeds"][0]
                assert "collecting" in next(
                    f.value or "" for f in sent.fields if f.name == "Build"
                )
                gate.set()
                await task

    async def test_a_probe_that_raises_degrades_to_its_own_blocks(
        self, mock_ctx: MagicMock
    ) -> None:
        """_safe_block guards each collector, so a raise here means the PROBE broke.
        It must cost its own blocks and nothing else."""
        mock_ctx.guild.voice_client = None

        async def boom() -> dict[str, list[str]]:
            raise RuntimeError("probe exploded")

        with patch.object(debug, "_build_blocks", boom):
            embed = await _snapshot(mock_ctx, self._operator())
        rendered = {f.name: (f.value or "") for f in embed.fields}
        assert "unavailable" in rendered["Build"]
        assert "shards" in rendered["Discord"]


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
    def _stats(self) -> ArchiveStats:
        return ArchiveStats(
            database_bytes=134217728,
            table_bytes=100663296,
            rows_estimate=412083,
            rejected_estimate=0,
            connections=3,
            max_connections=100,
            shared_buffers="128MB",
            cache_hit_ratio=0.994,
        )

    async def test_renders_the_archive_stats(self) -> None:
        archive = MagicMock()
        archive.stats = AsyncMock(return_value=self._stats())
        lines = "\n".join(await debug.postgres_lines(archive, archive_enabled=True))
        assert "db 128 MB" in lines
        assert "play_history 96 MB" in lines
        assert "~412083 plays · 0 rejected" in lines
        assert "3 / 100" in lines
        assert "hit rate 99.4%" in lines

    async def test_disabled_archive_reads_as_a_choice(self) -> None:
        """Mirrors -ping's OFF row: the ship default made this choice, and it must
        not read as a fault."""
        lines = await debug.postgres_lines(None, archive_enabled=False)
        assert lines == ["off (archive disabled)"]

    async def test_enabled_but_absent_archive_is_na(self) -> None:
        assert await debug.postgres_lines(None, archive_enabled=True) == ["n/a"]
