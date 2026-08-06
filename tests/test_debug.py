"""Debug mode's footer: what every reply grows while the mode is on.

Observation-only by rule, so what is asserted here is entirely what is DISPLAYED.
The seam that applies it to real command responses is tested in test_context.py.
"""

import asyncio
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import discord
import pytest
from opentelemetry import trace as trace_api

from src import debug


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
