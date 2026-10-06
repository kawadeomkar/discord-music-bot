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

import contextlib
import http.server
import socketserver
import subprocess
import threading
import time
from collections.abc import Iterator
from typing import Any, Optional

import discord
import pytest

from src.youtube import (
    YTDL,
    _ENCODE_BITRATE_CAP_KBPS,
    _OGG_HEADER_PACKETS,
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

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        type(self).seen_range = self.headers.get("Range")
        type(self).gets += 1
        mode = type(self).mode
        if mode == "refused":
            # A revoked URL: what YouTube returns once the signature expires.
            self.send_response(403)
            self.end_headers()
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
