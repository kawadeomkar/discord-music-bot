"""Tests for src/ytdl_formats.py — the `just ytdl-formats` diagnostic.

Only render() is covered, and deliberately: main() is one yt-dlp call with no
logic worth a double, while render() is where the report can quietly stop
reflecting what the bot does — it calls the real _mine_audio_candidates, so a
change to the mining rules shows up here rather than in a stale printout an
operator then trusts at the next yt-dlp bump."""

from typing import Any

from src.ytdl_formats import render


def _info(**overrides: Any) -> dict[str, Any]:
    audio = [
        {
            "format_id": "249",
            "url": "https://249",
            "vcodec": "none",
            "acodec": "opus",
            "abr": 46.0,
        },
        {
            "format_id": "251",
            "url": "https://251",
            "vcodec": "none",
            "acodec": "opus",
            "abr": 129.0,
        },
    ]
    base: dict[str, Any] = {
        "format_id": "251",
        "url": "https://251",
        "vcodec": "none",
        "acodec": "opus",
        "abr": 129.0,
        "formats": [
            {"format_id": "sb0", "url": "https://sb", "vcodec": "none", "acodec": None},
            *audio,
            {
                "format_id": "18",
                "url": "https://18",
                "vcodec": "avc1",
                "acodec": "mp4a.40.2",
                "height": 360,
            },
        ],
    }
    base.update(overrides)
    return base


class TestRender:
    def test_reports_the_selected_format_first(self) -> None:
        report = "\n".join(render(_info()))
        assert report.startswith("SELECTED (what the bot plays)")
        assert "251" in report.splitlines()[1]

    def test_reports_the_ladder_the_retry_would_actually_walk(self) -> None:
        """The point of the tool: not the raw format list, but what mining keeps —
        best first, storyboards and video rungs excluded."""
        report = render(_info())
        mined = report[
            report.index("MINED CANDIDATES (the retry ladder, best first)") :
        ]
        assert "0. 251" in mined[1]
        assert "1. 249" in mined[2]
        assert not any("sb0" in line for line in mined)

    def test_counts_the_audio_and_muxed_rungs(self) -> None:
        report = "\n".join(render(_info()))
        # Storyboards count as audio-less rungs here — the raw list is reported as
        # it is, and the mining section below it is where they are filtered.
        assert "AUDIO LADDER — 3 of 4 formats" in report
        assert "MUXED <=360p fallback rung: 1" in report

    def test_says_so_when_nothing_can_be_mined(self) -> None:
        """A flat playlist entry has no formats — an empty section would read like
        a mining bug rather than the wrong input."""
        report = "\n".join(render({"format_id": "x", "url": "https://x"}))
        assert "(none — no format list, or no stream URL)" in report
