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

- a refused URL (403) exits NON-ZERO, so discord.py stores an FFmpegProcessError
  and the retry fires;
- a connection that dies after the container header exits ZERO with no error, so
  the decision falls to `_drop_unplayable_stream_cache` instead — which is
  deliberate (see .claude/rules/playback.md: widening `stream_failed` onto the
  same evidence would eat a real history entry), and is pinned here so the choice
  stays a choice rather than an accident of ffmpeg's exit codes.

The two-pass `-ss` is deliberately NOT tested here. Whether ffmpeg turns an input
seek into an offset Range request depends on the file being large enough to be
worth one — a 29 KB sample is fetched whole either way — and a sample big enough
to show the difference takes longer to prove than the stall it prevents.

Needs ffmpeg on PATH — the same requirement `just run` already has. No network:
the server is local, and the sample is synthesised by ffmpeg at session scope.
"""

from __future__ import annotations

import contextlib
import http.server
import socketserver
import subprocess
import threading
from collections.abc import Iterator
from typing import Any, Optional

import discord
import pytest

from src.youtube import _OGG_HEADER_PACKETS
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

    def log_message(self, *_args: Any) -> None:  # noqa: D102 - quiet under pytest
        pass

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        type(self).seen_range = self.headers.get("Range")
        mode = type(self).mode
        if mode == "refused":
            # A revoked URL: what YouTube returns once the signature expires.
            self.send_response(403)
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
    with socketserver.TCPServer(("127.0.0.1", 0), _FailingHandler) as srv:
        thread = threading.Thread(target=srv.serve_forever, daemon=True)
        thread.start()
        _FailingHandler.url = f"http://127.0.0.1:{srv.server_address[1]}/a.webm"  # type: ignore[attr-defined]
        try:
            yield _FailingHandler
        finally:
            srv.shutdown()
            thread.join(timeout=5)


def _drain(url: str) -> tuple[int, Optional[int], Optional[Exception]]:
    """Read a source to exhaustion the way discord.py's AudioPlayer does.

    Returns (packets, ffmpeg's exit code, the error discord.py stored) — the
    three values `stream_failed` is computed from, before any of our own code
    interprets them."""
    source = discord.FFmpegOpusAudio(url, options="-vn")
    process = source._process
    packets = 0
    try:
        while source.read():
            packets += 1
        # read() checks the return code only when it hands back an empty packet,
        # which is exactly the call that ended the loop above.
        return packets, process.poll(), getattr(source, "_current_error", None)
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
        assert code not in (0, None), "ffmpeg reported success for a refused URL"
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
