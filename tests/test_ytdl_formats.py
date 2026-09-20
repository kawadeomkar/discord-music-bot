"""Tests for src/ytdl_formats.py — the `just ytdl-formats` diagnostic.

render() is where the report can quietly stop reflecting what the bot does — it
mines and reconstructs with the real functions, so a change to those rules shows up
here rather than in a stale printout an operator trusts at the next yt-dlp bump.

main() is covered only for the one decision it makes: which entry of a search result
to report on. It has to make the SAME choice yt_source does, and the way that breaks
is someone re-inlining the rule instead of calling the shared picker."""

from typing import Any
from unittest.mock import MagicMock, patch

from src.ytdl_formats import main, render


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
        """A flat playlist entry has no formats. Rung 0 still exists (it is the
        top-level URL), so what is missing is the fallbacks — and saying so beats an
        empty section, which reads like a mining bug rather than the wrong input."""
        report = "\n".join(render({"format_id": "x", "url": "https://x"}))
        assert "(no alternatives — no format list, or no stream URL)" in report


class TestMainEntrySelection:
    """main() reports on the entry the bot would actually play."""

    def _run(self, raw: Any) -> tuple[int, MagicMock]:
        ydl = MagicMock()
        ydl.extract_info.return_value = raw
        with (
            patch("src.ytdl_formats.sys.argv", ["ytdl-formats", "some search"]),
            patch("src.ytdl_formats.yt_dlp.YoutubeDL", return_value=ydl),
            patch("builtins.print") as printer,
        ):
            return main(), printer

    def test_it_reports_the_entry_yt_source_would_pick(self) -> None:
        """The entry carrying a stream URL, not merely the first one — via the shared
        picker, because a re-inlined copy of that rule drifts and this tool then
        answers for a song the bot does not choose."""
        code, printer = self._run(
            {
                "entries": [
                    {"_type": "video", "id": "no-url", "format_id": "a"},
                    {
                        "_type": "video",
                        "id": "playable",
                        "url": "https://b",
                        "format_id": "251",
                    },
                ]
            }
        )
        assert code == 0
        assert "251" in "\n".join(str(c.args[0]) for c in printer.call_args_list)

    def test_a_url_less_entry_is_still_reported_on(self) -> None:
        """The case that separates the shared picker from the obvious re-inlining of
        its rule: with no entry carrying a stream URL, yt_source still falls back to
        the first non-playlist entry and plays it, so the diagnostic must report that
        entry rather than declaring the search unplayable."""
        code, printer = self._run(
            {
                "entries": [
                    {"_type": "playlist", "id": "p"},
                    {"_type": "video", "id": "first", "format_id": "fallback-rung"},
                ]
            }
        )
        assert code == 0
        assert "fallback-rung" in "\n".join(
            str(c.args[0]) for c in printer.call_args_list
        )

    def test_an_all_playlist_result_is_reported_not_crashed(self) -> None:
        """This used to IndexError on entries[0] of an empty playable list."""
        code, _ = self._run({"entries": [{"_type": "playlist"}, None]})
        assert code == 1
