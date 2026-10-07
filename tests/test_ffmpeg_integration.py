"""Opt-in tier: what a REAL ffmpeg does to a stream that fails, and how
discord.py reports it.

`stream_failed = not song.produced_audio and play_error[0] is not None` decides
whether the retry ladder runs. Both halves are facts about ffmpeg's exit code and
discord.py's `_check_process_returncode`, and the default suite asserts them by
INJECTION: it hands the loop a `produced_audio` and an error and checks what it
does next. Nothing there notices when ffmpeg starts exiting 0 where it used to
exit non-zero, or when a container's header packet count moves — and either one
silently turns the retry off in production while the suite stays green.

So this tier spawns the real binary against a local HTTP server that fails on
purpose, and pins the two numbers the decision is built on:

- a refused URL (403) exits NON-ZERO, so discord.py turns that into an
  FFmpegProcessError and the retry can fire;
- a connection that dies after the container header exits ZERO with no error, so
  the decision falls to `_drop_unplayable_stream_cache` instead — which is
  deliberate (see .claude/rules/playback.md: widening `stream_failed` onto the
  same evidence would eat a real history entry), and is pinned here so the choice
  stays a choice rather than an accident of ffmpeg's exit codes.

A fourth fact is the loudness setting's ceiling: `alimiter` is transparent below
it only with `latency=1`, and the difference is in the samples, not in any exit
code. That one measures the filter directly rather than through discord.py,
because what is under test is the filter.

A third fact lives here for the same reason: which Opus mode the encode path
leaves libopus in. discord.py asks for in-band FEC on every spawn, which libopus
answers by abandoning CELT, and nothing but the bitstream shows it — the argv is
accepted, the exit code is 0, and the audio plays. `YTDL.FFMPEG_OPTS` overrides
the pair; the control test reads the TOC bytes under discord.py's bare defaults
so the override's reason stays measurable. See docs/ARCHITECTURE.md#encoder-mode.

The two-pass `-ss` is deliberately NOT tested here. Whether ffmpeg turns an input
seek into an offset Range request depends on the file being large enough to be
worth one — a 29 KB sample is fetched whole either way — and a sample big enough
to show the difference takes longer to prove than the stall it prevents.

What it does NOT assert is that discord.py catches the exit code in time. ffmpeg
closes stdout before it is reaped, so `read()`'s own `poll()` can return None and
store nothing — the window playback.md names, whose backstop is
`_drop_unplayable_stream_cache`. That is a race, not a fact, and asserting it is
how the first version of this file passed on a laptop and failed on a runner.

Needs ffmpeg on PATH — the same requirement `just run` already has, though note
GitHub's runner image does NOT ship it, so the CI job installs it. No network:
the server is local, and the sample is synthesised by ffmpeg at session scope.
"""

from __future__ import annotations

import array
import contextlib
import http.server
import math
import re
import signal
import socketserver
import subprocess
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import discord
import pytest

from src import youtube
from src.youtube import (
    YTDL,
    Loudness,
    _ENCODE_BITRATE_CAP_KBPS,
    _OGG_HEADER_PACKETS,
    _PEAK_LIMITER,
    measure_loudness,
)
from tests.helpers import tier_enabled

pytestmark = [
    pytest.mark.ffmpeg,
    pytest.mark.skipif(
        not tier_enabled("RUN_FFMPEG_TESTS"),
        reason="opt-in tier: RUN_FFMPEG_TESTS=1 (needs ffmpeg on PATH)",
    ),
]

# Long enough to deliver many packets past the header, short enough that a
# failing test does not sit here: 3s is ~150 20ms frames.
_SAMPLE_SECS = 3
# How long "half_then_stall" holds its connection. The single-threaded server cannot
# shut down until it returns, so it is short; the scan's deadline is shorter still.
_STALL_SECS = 3.0


@pytest.fixture(scope="session")
def opus_sample(tmp_path_factory: pytest.TempPathFactory) -> bytes:
    """A real opus/webm file, built by the same ffmpeg under test.

    Synthesised rather than committed: a binary fixture would be one more thing
    to keep in step with the codec claims in youtube.py, and this proves the
    local ffmpeg can produce what it is about to be asked to read."""
    path = tmp_path_factory.mktemp("ffmpeg") / "sample.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={_SAMPLE_SECS}",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            str(path),
            "-y",
        ],
        check=True,
    )
    return path.read_bytes()


class _FailingHandler(http.server.BaseHTTPRequestHandler):
    """Serves the sample, or one of the ways YouTube stops serving it."""

    payload: bytes = b""
    mode: str = "ok"
    seen_range: Optional[str] = None
    gets: int = 0
    # (first byte, bytes delivered) of every bounded range served, in order.
    served: list[tuple[int, int]] = []

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - quiet under pytest
        pass

    def _serve_half(self) -> None:
        """Open, deliver half the sample, then let the connection die."""
        payload = type(self).payload
        self.send_response(200)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(payload[: len(payload) // 2])
        self.wfile.flush()
        self.close_connection = True

    def _serve_range(self) -> None:
        """Honour the Range the reconnect asks for, which is what makes the
        recovered stream the same audio rather than merely the same length."""
        payload = type(self).payload
        raw = self.headers.get("Range") or ""
        start = int(raw.split("=")[1].split("-")[0]) if "=" in raw else 0
        self.send_response(206 if start else 200)
        self.send_header(
            "Content-Range", f"bytes {start}-{len(payload) - 1}/{len(payload)}"
        )
        self.send_header("Content-Length", str(len(payload) - start))
        self.end_headers()
        self.wfile.write(payload[start:])

    def _serve_bounded(self, first: int, last: int) -> None:
        """One request of a ranged scan: that slice as a 206, or half of it and a
        dead connection."""
        mode = type(self).mode
        payload = type(self).payload
        body = payload[first : last + 1]
        self.send_response(206)
        self.send_header("Content-Range", f"bytes {first}-{last}/{len(payload)}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if mode == "ranges_die" or (mode == "range_dies_once" and type(self).gets == 2):
            body = body[: len(body) // 2]
            self.close_connection = True
        type(self).served.append((first, len(body)))
        self.wfile.write(body)
        self.wfile.flush()

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        type(self).seen_range = self.headers.get("Range")
        type(self).gets += 1
        mode = type(self).mode
        if mode == "refused":
            # A revoked URL: what YouTube returns once the signature expires.
            self.send_response(403)
            self.end_headers()
            return
        # ffmpeg only ever asks open-ended (`bytes=N-`); a bounded range is the
        # scan's own fetch.
        bounded = re.fullmatch(r"bytes=(\d+)-(\d+)", self.headers.get("Range") or "")
        if bounded:
            self._serve_bounded(int(bounded[1]), int(bounded[2]))
            return
        if mode == "dead_then_503_then_ok":
            # A transient CDN failure, and the shape the reconnect flag exists for:
            # the connection dies mid-stream, two retries are refused, the fourth
            # request is honoured as a Range. Content-Length on the 503 is what
            # makes it a well-formed HTTP error rather than another truncation —
            # a plain `-reconnect` already retries the latter.
            if type(self).gets == 1:
                self._serve_half()
                return
            if type(self).gets in (2, 3):
                self.send_response(503)
                self.send_header("Content-Length", "0")
                self.end_headers()
                self.close_connection = True
                return
            self._serve_range()
            return
        if mode == "dead_then_403":
            # The same death, never answered. The retry flag must not turn this
            # into a wait: `stream_failed` and the cache drop decide it as before.
            if type(self).gets == 1:
                self._serve_half()
                return
            self.send_response(403)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        if mode == "half_then_stall":
            # A fetch that stops making progress with the connection still open:
            # what a throttled CDN looks like to a scan at its deadline.
            payload = type(self).payload
            self.send_response(200)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload[: len(payload) // 2])
            self.wfile.flush()
            time.sleep(_STALL_SECS)
            self.close_connection = True
            return
        if mode == "header_then_dead":
            # Opens, delivers the container header, then the connection dies.
            self.send_response(200)
            self.send_header("Content-Length", str(len(type(self).payload)))
            self.end_headers()
            self.wfile.write(type(self).payload[:512])
            self.wfile.flush()
            self.close_connection = True
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(type(self).payload)))
        self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self.wfile.write(type(self).payload)


@pytest.fixture
def server(opus_sample: bytes) -> Iterator[type[_FailingHandler]]:
    _FailingHandler.payload = opus_sample
    _FailingHandler.mode = "ok"
    _FailingHandler.seen_range = None
    _FailingHandler.gets = 0
    _FailingHandler.served = []
    with socketserver.TCPServer(("127.0.0.1", 0), _FailingHandler) as srv:
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        _FailingHandler.url = f"http://127.0.0.1:{srv.server_address[1]}/a.webm"  # type: ignore[attr-defined]
        try:
            yield _FailingHandler
        finally:
            srv.shutdown()
            thread.join(timeout=5)


# A child that has closed stdout is about to exit; this only bounds the wait so
# a hung ffmpeg fails the test rather than the run.
_EXIT_TIMEOUT_SECS = 10


def _packets(url: str, before_options: Optional[str] = None) -> list[bytes]:
    """Every Opus packet a source yields, in order. The packets rather than their
    count, because a reconnect that overlaps or skips a byte range still delivers
    the right NUMBER of them."""
    source = discord.FFmpegOpusAudio(url, before_options=before_options, options="-vn")
    process = source._process
    packets: list[bytes] = []
    try:
        while packet := source.read():
            packets.append(bytes(packet))
        process.wait(timeout=_RECONNECT_TIMEOUT_SECS)
        return packets
    finally:
        source.cleanup()
        for pipe in (process.stdout, process.stderr, process.stdin):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


def _drain(
    url: str, before_options: Optional[str] = None
) -> tuple[int, int, Optional[Exception]]:
    """Read a source to exhaustion the way discord.py's AudioPlayer does, then
    let the child exit and ask discord.py what it makes of the exit code.

    Returns (packets, ffmpeg's final exit code, the error discord.py stored).

    `wait()`, not the `poll()` that `read()` itself uses: ffmpeg closes stdout
    before it is reaped, so polling the instant the loop ends reads None on a
    fast host and the real code on a slow one — this asserted `code == 0` on a
    laptop and `code is None` on a CI runner. That window is real and the
    production code owns it (playback.md: `_drop_unplayable_stream_cache` is its
    backstop), but it is a race, and what this tier exists to pin is the exit
    code itself. `_check_process_returncode` is then called explicitly, which is
    what `read()` would have done had the child been reaped in time."""
    source = discord.FFmpegOpusAudio(url, before_options=before_options, options="-vn")
    process = source._process
    packets = 0
    try:
        while source.read():
            packets += 1
        code = process.wait(timeout=_EXIT_TIMEOUT_SECS)
        source._check_process_returncode()
        return packets, code, getattr(source, "_current_error", None)
    finally:
        source.cleanup()
        # cleanup() kills the child but leaves its pipes open, and this suite
        # runs with filterwarnings=error, where a ResourceWarning is a failure.
        for pipe in (process.stdout, process.stderr, process.stdin):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


class TestWhatFFmpegReportsForAFailedStream:
    """The two numbers `stream_failed` is built on, measured rather than assumed."""

    def test_a_healthy_stream_produces_audio_and_exits_clean(
        self, server: type[_FailingHandler]
    ) -> None:
        packets, code, error = _drain(server.url)  # type: ignore[attr-defined]
        assert packets > _OGG_HEADER_PACKETS, "no audio past the container header"
        assert code == 0
        assert error is None

    def test_a_refused_url_exits_non_zero_so_the_retry_can_fire(
        self, server: type[_FailingHandler]
    ) -> None:
        """The retry ladder's entire trigger. If ffmpeg ever starts exiting 0 here,
        `play_error` is None, `stream_failed` is False, and a revoked URL silently
        ends the song instead of being retried on a fresh one."""
        server.mode = "refused"
        packets, code, error = _drain(server.url)  # type: ignore[attr-defined]

        assert packets <= _OGG_HEADER_PACKETS, "a 403 must not look like audio"
        assert code != 0, "ffmpeg reported success for a refused URL"
        assert isinstance(error, discord.errors.ClientException), (
            "discord.py did not store an error, so `after` receives None and "
            "stream_failed is False"
        )

    def test_a_header_then_dead_connection_exits_zero_and_is_the_cache_drop(
        self, server: type[_FailingHandler]
    ) -> None:
        """Deliberately NOT the retry path. ffmpeg exits 0 here, so `play_error`
        is None and the loop falls to `_drop_unplayable_stream_cache` — see
        .claude/rules/playback.md. Pinned so the split stays a decision: if this
        ever starts exiting non-zero, the case silently becomes a retry, which
        the rule file says would eat a real history entry."""
        server.mode = "header_then_dead"
        packets, code, error = _drain(server.url)  # type: ignore[attr-defined]

        assert packets <= _OGG_HEADER_PACKETS, "delivered audio it should not have"
        assert code == 0
        assert error is None

    def test_the_header_packet_threshold_is_what_a_container_actually_emits(
        self, server: type[_FailingHandler]
    ) -> None:
        """`_OGG_HEADER_PACKETS` is an empirical constant: it is the number of
        packets a stream yields before any audio, and `produced_audio` subtracts
        it. Too low and a dead stream reads as having played; too high and a very
        short song reads as dead."""
        server.mode = "header_then_dead"
        header_only, _, _ = _drain(server.url)  # type: ignore[attr-defined]
        assert header_only == _OGG_HEADER_PACKETS, (
            f"a stream that delivered no audio yielded {header_only} packets, but "
            f"_OGG_HEADER_PACKETS is {_OGG_HEADER_PACKETS}"
        )


# RFC 6716 §3.1: TOC configs 0-11 are SILK, 12-15 hybrid, 16-31 CELT.
_CELT_CONFIG_FLOOR = 16
# Enough to see a mode hold rather than a first-frame choice; ~1s of the 3s sample.
_MODE_SAMPLE_PACKETS = 50


def _audio_bytes(url: str, *, bitrate: Optional[int] = None) -> int:
    """How many bytes of Opus the player reads for a fixed number of packets, at
    `bitrate` kbps. Packets, not file size, because the Ogg framing is not the wire."""
    source = discord.FFmpegOpusAudio(
        url, options=YTDL.FFMPEG_OPTS["options"], bitrate=bitrate
    )
    process = source._process
    try:
        for _ in range(_OGG_HEADER_PACKETS):
            source.read()
        return sum(len(source.read()) for _ in range(_MODE_SAMPLE_PACKETS))
    finally:
        source.cleanup()
        for pipe in (process.stdout, process.stderr, process.stdin):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


class TestTheChannelBitrateIsSpent:
    """`_encode_bitrate_kbps` decides the number; this is the only place that reads
    what libopus does with it. The argv is asserted in the default tier against a
    patched `FFmpegOpusAudio`, which cannot see the bitstream.
    See docs/ARCHITECTURE.md#encoder-mode."""

    def test_a_raised_bitrate_reaches_the_bitstream(
        self, server: type[_FailingHandler]
    ) -> None:
        """Monotonic rather than a fixed figure: libopus runs unconstrained VBR, so
        what a target delivers depends on the content, but more target is more bits
        for the same input."""
        url = server.url  # pyright: ignore[reportAttributeAccessIssue]
        default = _audio_bytes(url)
        raised = _audio_bytes(url, bitrate=_ENCODE_BITRATE_CAP_KBPS)
        assert raised > default, (default, raised)


# Production's `before_options` minus the flag under test, derived rather than
# written out: a change to the base flags would otherwise leave the control
# quietly standing in for something that is no longer the baseline.
_BASE_RECONNECT = YTDL.FFMPEG_OPTS["before_options"].replace(
    " -reconnect_on_http_error 5xx", ""
)


def _audio_tocs(url: str, options: str) -> list[int]:
    """The TOC config of the first _MODE_SAMPLE_PACKETS audio packets, read the
    way the player reads them, past the two Ogg header packets."""
    source = discord.FFmpegOpusAudio(url, options=options)
    process = source._process
    try:
        for _ in range(_OGG_HEADER_PACKETS):
            source.read()
        tocs = []
        for _ in range(_MODE_SAMPLE_PACKETS):
            packet = source.read()
            # read() answers b"" once the stream is spent; [0] on that is an
            # IndexError that says nothing about the sample being too short.
            assert packet, f"stream ended after {len(tocs)} of {_MODE_SAMPLE_PACKETS}"
            tocs.append(packet[0] >> 3)
        return tocs
    finally:
        source.cleanup()
        for pipe in (process.stdout, process.stderr, process.stdin):
            if pipe is not None:
                with contextlib.suppress(OSError):
                    pipe.close()


class TestEncoderMode:
    """Which Opus mode the re-encode path leaves libopus in, measured rather than
    assumed. See docs/ARCHITECTURE.md#encoder-mode."""

    def test_the_shipped_options_hold_celt(self, server: type[_FailingHandler]) -> None:
        tocs = _audio_tocs(server.url, YTDL.FFMPEG_OPTS["options"])  # pyright: ignore[reportAttributeAccessIssue]
        assert all(t >= _CELT_CONFIG_FLOOR for t in tocs), tocs

    def test_discord_py_defaults_alone_leave_celt(
        self, server: type[_FailingHandler]
    ) -> None:
        """The control. When this starts seeing CELT, the override no longer
        earns its place and the encoder-mode section can shrink."""
        tocs = _audio_tocs(server.url, "-vn")  # pyright: ignore[reportAttributeAccessIssue]
        assert all(t < _CELT_CONFIG_FLOOR for t in tocs), tocs


# ffmpeg's `sine` source generates at about -21 dBFS, which is already comfortably
# under the limiter's 0.891 ceiling; +24 dB puts it over, and clipping.
_OVER_THE_CEILING_DB = 24
# One step of 16-bit quantisation. The limiter's pass-through gain is 1.0 to within
# float rounding (7.5e-09 on ffmpeg 9.0.2, four thousand times under this step), so a
# quantised difference lands here or nowhere.
_ONE_LSB = 1
# Far above _ONE_LSB and far below the ~3300 the 5 ms shift measures, so the control
# fails on that shift rather than on a build's rounding.
_A_REAL_CHANGE = 100
# 0.891 of full scale, plus a decibel of slack for the encode either side of it.
_CEILING_SAMPLE = int(0.891 * 32768)
_ENCODE_SLACK = 400


def _sine(directory: Path, *, gain_db: int = 0) -> str:
    """A sample built like the tier's own, at `gain_db` relative to it."""
    path = directory / f"sine{gain_db}.webm"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={_SAMPLE_SECS}",
            "-af",
            f"volume={gain_db}dB",
            "-c:a",
            "libopus",
            "-f",
            "webm",
            str(path),
            "-y",
        ],
        check=True,
    )
    return str(path)


def _decoded(path: str, *filters: str) -> array.array[int]:
    """`path` decoded to 48 kHz stereo 16-bit, through `filters` when given."""
    chain = ["-filter:a", ",".join(filters)] if filters else []
    result = subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-vn",
            *chain,
            "-f",
            "s16le",
            "-ar",
            "48000",
            "-ac",
            "2",
            "-",
        ],
        check=True,
        stdout=subprocess.PIPE,
    )
    samples: array.array[int] = array.array("h")
    samples.frombytes(result.stdout)
    return samples


def _worst_difference(a: array.array[int], b: array.array[int]) -> int:
    assert len(a) == len(b), f"lengths differ: {len(a)} vs {len(b)}"
    return max((abs(x - y) for x, y in zip(a, b)), default=0)


class TestTheLoudnessCeilingIsTransparent:
    """`peak` and `normalize` both end in `alimiter`, which every song they touch
    passes through. A limiter that alters a song it was not meant to cap is a
    quality regression with no symptom: no error, no exit code, and audio that
    still plays. See docs/ARCHITECTURE.md#loudness-normalization."""

    def test_the_filtergraph_alone_is_bit_exact(self, tmp_path: Path) -> None:
        """The baseline the next two are read against: inserting a filtergraph at
        all costs nothing, so any difference below belongs to the limiter."""
        sample = _sine(tmp_path)
        assert _worst_difference(_decoded(sample), _decoded(sample, "anull")) == 0

    def test_below_the_ceiling_the_shipped_limiter_changes_nothing(
        self, tmp_path: Path
    ) -> None:
        sample = _sine(tmp_path)
        assert (
            _worst_difference(_decoded(sample), _decoded(sample, _PEAK_LIMITER))
            <= _ONE_LSB
        )

    def test_without_latency_1_every_sample_moves(self, tmp_path: Path) -> None:
        """The control, and the whole reason that option is in the string:
        alimiter's default reports its 5 ms lookahead as latency, which ffmpeg
        answers by delaying the stream. Nothing else in the chain would show it."""
        unlatched = _PEAK_LIMITER.replace(":latency=1", "")
        assert unlatched != _PEAK_LIMITER, "the shipped filter no longer sets latency"
        sample = _sine(tmp_path)
        assert (
            _worst_difference(_decoded(sample), _decoded(sample, unlatched))
            > _A_REAL_CHANGE
        )

    def test_a_peak_over_the_ceiling_is_brought_under_it(self, tmp_path: Path) -> None:
        """The other half: a limiter that caps nothing is not a ceiling."""
        hot = _sine(tmp_path, gain_db=_OVER_THE_CEILING_DB)
        assert max(abs(s) for s in _decoded(hot)) > _CEILING_SAMPLE
        capped = _decoded(hot, _PEAK_LIMITER)
        assert max(abs(s) for s in capped) <= _CEILING_SAMPLE + _ENCODE_SLACK


class TestMeasureLoudnessReadsTheRealBinary:
    """`normalize` decides a song's gain from what ffmpeg prints, and the unit tests
    parse a captured block. This is what notices when a build stops printing it in
    that shape: the scan would then return None for every song and the setting would
    quietly do nothing but re-encode."""

    async def test_the_tiers_sample_measures(
        self, server: type[_FailingHandler]
    ) -> None:
        measured = await measure_loudness(server.url, "https://yt.com/v=tier", None)  # pyright: ignore[reportAttributeAccessIssue]

        assert measured is not None, "ebur128 printed no summary this parser could read"
        assert math.isfinite(measured.i) and math.isfinite(measured.peak)
        # The sine is about -21 dBFS, so both readings land well under 0.
        assert -60.0 < measured.i < 0.0
        assert -60.0 < measured.peak <= 0.0

    async def test_a_url_that_serves_nothing_is_not_a_level(
        self, server: type[_FailingHandler]
    ) -> None:
        """A refused URL has to answer None rather than raise: the scan runs inside
        the prefetch, where a level is an improvement to a play and not a
        precondition for one."""
        server.mode = "refused"
        assert await measure_loudness(server.url, "https://yt.com/v=gone", None) is None  # pyright: ignore[reportAttributeAccessIssue]


class TestALongSongIsMeasuredFromItsStart:
    """A song past the cap is measured from its first `_SCAN_MAX_AUDIO_SECS`, which
    ffmpeg's own `-t` has to honour and its closing line has to report."""

    async def test_the_reading_covers_the_cap(
        self, server: type[_FailingHandler], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(youtube, "_SCAN_MAX_AUDIO_SECS", 1.0)
        measured = await measure_loudness(
            server.url,  # pyright: ignore[reportAttributeAccessIssue]
            "https://yt.com/v=capped",
            None,
            duration=_SAMPLE_SECS,
        )

        assert measured is not None and measured.covered_secs is not None
        assert measured.covered_secs == pytest.approx(1.0, abs=0.1)


class TestAScanStoppedAtItsDeadline:
    """SIGINT, not kill: ffmpeg stops reading and prints the summary of the audio
    it has, with the closing line that says how much that was. The unit tests fake
    both halves of that; this is the binary doing it."""

    async def test_what_it_read_is_the_reading(
        self, server: type[_FailingHandler], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server.mode = "half_then_stall"
        monkeypatch.setattr(youtube, "_SCAN_DEADLINE_SECS", 1.0)
        monkeypatch.setattr(youtube, "_PARTIAL_MIN_SECS", 0.5)
        started = time.monotonic()

        measured = await measure_loudness(
            server.url,  # pyright: ignore[reportAttributeAccessIssue]
            "https://yt.com/v=stalled",
            None,
            duration=_SAMPLE_SECS,
        )

        assert time.monotonic() - started < _STALL_SECS
        assert measured is not None and measured.covered_secs is not None
        assert 0.5 <= measured.covered_secs < _SAMPLE_SECS
        assert -60.0 < measured.i < 0.0

    def test_one_signal_does_not_break_a_stalled_read(
        self, server: type[_FailingHandler]
    ) -> None:
        """The control, and the reason `_interrupt` sends two: the first is read
        between packets, and a fetch stalled inside a read never gets there."""
        server.mode = "half_then_stall"
        child = subprocess.Popen(
            ["ffmpeg", "-nostats", "-loglevel", "info", "-i", server.url]  # pyright: ignore[reportAttributeAccessIssue]
            + ["-vn", "-af", "ebur128=framelog=quiet:peak=sample", "-f", "null", "-"],
            stderr=subprocess.PIPE,
        )
        try:
            time.sleep(0.5)
            child.send_signal(signal.SIGINT)
            time.sleep(0.75)
            assert child.poll() is None, "one SIGINT broke the read after all"
            child.send_signal(signal.SIGINT)
            _, printed = child.communicate(timeout=_STALL_SECS)
        finally:
            if child.poll() is None:
                child.kill()
                child.communicate()
        assert youtube.parse_ebur128_summary(printed.decode()) is not None


class TestAnHlsPlaylistIsScanned:
    """SoundCloud's serves are HLS playlists of fMP4 segments, a different demuxer
    from the webm every other test here reads — and the scan hands it the play
    path's reconnect flags, which it must accept."""

    @pytest.fixture
    def playlist(self, tmp_path: Path) -> Iterator[str]:
        """A VOD playlist of the tier's tone in 1 s fMP4 segments, served locally."""
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                f"sine=frequency=440:duration={_SAMPLE_SECS}",
                "-c:a",
                "aac",
                "-f",
                "hls",
                "-hls_time",
                "1",
                "-hls_playlist_type",
                "vod",
                "-hls_segment_type",
                "fmp4",
                str(tmp_path / "playlist.m3u8"),
            ],
            check=True,
        )

        class Quiet(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, directory=str(tmp_path), **kwargs)

            def log_message(self, *_args: Any) -> None:
                pass

        with socketserver.TCPServer(("127.0.0.1", 0), Quiet) as srv:
            thread = threading.Thread(target=srv.serve_forever, daemon=True)
            thread.start()
            try:
                yield f"http://127.0.0.1:{srv.server_address[1]}/playlist.m3u8"
            finally:
                srv.shutdown()
                thread.join(timeout=5)

    async def test_it_measures(self, playlist: str) -> None:
        measured = await measure_loudness(playlist, "https://sc.com/a/hls", None)
        assert measured is not None
        assert -60.0 < measured.i < 0.0


class TestALongSongIsScannedInRanges:
    """A googlevideo file over 10 MB is fetched by the bot in ranges and piped to
    ffmpeg, because ffmpeg's single request for it is throttled past any scan
    timeout. What is under test is the seam the unit tests fake on both sides: real
    aiohttp reading real ranges into a real ffmpeg's stdin.
    See docs/ARCHITECTURE.md#loudness-normalization."""

    _RANGES = 4

    @pytest.fixture(autouse=True)
    def _as_a_long_googlevideo_song(
        self, monkeypatch: pytest.MonkeyPatch, server: type[_FailingHandler]
    ) -> None:
        """The host gate is the one part a local server cannot meet."""
        size = len(server.payload)
        monkeypatch.setattr(youtube, "_ranged_scan_size", lambda _url: size)
        monkeypatch.setattr(youtube, "_SCAN_RANGE_BYTES", -(-size // self._RANGES))

    @staticmethod
    async def _scan(server: type[_FailingHandler], page: str) -> Optional[Loudness]:
        return await measure_loudness(server.url, page, None)  # pyright: ignore[reportAttributeAccessIssue]

    async def _whole(self, server: type[_FailingHandler]) -> Optional[Loudness]:
        """The level ffmpeg reads off this sample in one request of its own."""
        with pytest.MonkeyPatch.context() as patched:
            patched.setattr(youtube, "_ranged_scan_size", lambda _url: None)
            whole = await self._scan(server, "https://yt.com/v=whole")
        server.gets = 0
        return whole

    @staticmethod
    def _assert_every_byte_arrived_once(server: type[_FailingHandler]) -> None:
        """Read off the server rather than the level: ffmpeg resyncs past a
        duplicated or missing stretch of this sample and measures it the same."""
        position = 0
        for first, delivered in server.served:
            assert first == position
            position += delivered
        assert position == len(server.payload)

    async def test_it_measures_what_one_request_does(
        self, server: type[_FailingHandler]
    ) -> None:
        whole = await self._whole(server)

        ranged = await self._scan(server, "https://yt.com/v=ranged")

        assert server.gets == self._RANGES
        self._assert_every_byte_arrived_once(server)
        assert ranged is not None and ranged == whole

    async def test_a_capped_song_is_fed_only_its_share(
        self, server: type[_FailingHandler], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Closing stdin early is ffmpeg's end of input, so it measures the
        prefix it was fed rather than failing."""
        monkeypatch.setattr(youtube, "_SCAN_MAX_AUDIO_SECS", _SAMPLE_SECS / 2)

        measured = await measure_loudness(
            server.url,  # pyright: ignore[reportAttributeAccessIssue]
            "https://yt.com/v=prefix",
            None,
            duration=_SAMPLE_SECS,
        )

        delivered = sum(n for _, n in server.served)
        assert delivered < len(server.payload)
        assert measured is not None and measured.covered_secs is not None
        assert 0.5 < measured.covered_secs < _SAMPLE_SECS

    async def test_a_range_that_dies_is_resumed_where_it_stopped(
        self, server: type[_FailingHandler]
    ) -> None:
        whole = await self._whole(server)
        server.mode = "range_dies_once"

        ranged = await self._scan(server, "https://yt.com/v=resumed")

        assert server.gets == self._RANGES + 1
        self._assert_every_byte_arrived_once(server)
        assert ranged is not None and ranged == whole

    async def test_ranges_that_keep_dying_are_not_a_level(
        self, server: type[_FailingHandler]
    ) -> None:
        """ffmpeg exits 0 with a summary of the audio that did arrive, so the feed's
        failure is the only thing that says the song was not all read."""
        server.mode = "ranges_die"

        assert await self._scan(server, "https://yt.com/v=dying") is None
        assert server.gets == youtube._SCAN_FETCH_RETRIES + 1

    async def test_a_refused_range_is_not_asked_for_again(
        self, server: type[_FailingHandler]
    ) -> None:
        server.mode = "refused"

        assert await self._scan(server, "https://yt.com/v=gone") is None
        assert server.gets == 1


# The reconnect backoff is 0 + 1 + 3 s before -reconnect_delay_max refuses a 7,
# so a recovering stream needs several seconds and a stalled one gives up inside ~4.
_RECONNECT_TIMEOUT_SECS = 40


class TestAMidSongReconnect:
    """A song whose connection dies mid-stream. `-reconnect` alone retries a
    truncated stream but not a well-formed HTTP error, so a transient 503 used to
    end the song wherever it died — exit 0, no error, no retry, and nothing in the
    logs a listener could connect to the audio stopping.
    See docs/ARCHITECTURE.md#mid-song-reconnects."""

    def test_a_transient_503_recovers_the_whole_song(
        self, server: type[_FailingHandler]
    ) -> None:
        healthy = _packets(server.url)  # pyright: ignore[reportAttributeAccessIssue]
        server.mode = "dead_then_503_then_ok"
        server.gets = 0

        recovered = _packets(
            server.url,  # pyright: ignore[reportAttributeAccessIssue]
            YTDL.FFMPEG_OPTS["before_options"],
        )

        # Byte-identical, not merely the same length: the reconnect resumes by
        # Range, and an overlapping or short range would still count right.
        assert recovered == healthy
        assert server.gets == 4, "the 503s were not retried"

    def test_without_the_flag_the_song_ends_where_it_died(
        self, server: type[_FailingHandler]
    ) -> None:
        """The control, and the measurement that earns the flag: half a song,
        reported as a clean end."""
        healthy = _packets(server.url)  # pyright: ignore[reportAttributeAccessIssue]
        server.mode = "dead_then_503_then_ok"
        server.gets = 0

        truncated = _packets(
            server.url,  # pyright: ignore[reportAttributeAccessIssue]
            _BASE_RECONNECT,
        )

        assert len(truncated) < len(healthy) / 2 + 2
        # What it did deliver is still the right audio; it just stops.
        assert truncated[:-1] == healthy[: len(truncated) - 1]

    def test_a_death_that_is_never_answered_still_ends_the_song(
        self, server: type[_FailingHandler]
    ) -> None:
        """The flag must not turn an unrecoverable death into a wait. ffmpeg does
        not retry a 403 when the list is `5xx`, so this decides exactly as it did
        before: exit 0, no error, and the loop falls to
        `_drop_unplayable_stream_cache` rather than the retry ladder."""
        server.mode = "dead_then_403"
        server.gets = 0
        started = time.monotonic()

        packets, code, error = _drain(
            server.url,  # pyright: ignore[reportAttributeAccessIssue]
            YTDL.FFMPEG_OPTS["before_options"],
        )

        assert (code, error) == (0, None)
        assert packets > _OGG_HEADER_PACKETS, "delivered nothing at all"
        assert time.monotonic() - started < 5.0, "waited on a death nothing will answer"
