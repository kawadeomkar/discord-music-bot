"""What would the bot actually play? — `just ytdl-formats <url>`.

A diagnostic, not part of the running bot. It runs the real `_YTDL_STREAM_OPTS`
against one URL and prints three things: the format yt-dlp selected, the full
audio ladder it chose from, and the fallback ladder `_mine_audio_candidates`
would keep. That last one is the production decision — everything the stream
retry can fall back to comes from it.

Run it at every yt-dlp bump. The claims this repo makes about format selection
(bestaudio already picks the ladder top; opus beats AAC at equal quality tier;
there is no higher rung without Premium cookies) are empirical, and YouTube and
yt-dlp both move. See docs/ARCHITECTURE.md#stream-retry-ladder.
"""

import copy
import sys
from typing import Any, cast

import yt_dlp

from src.youtube import (
    YTDL,
    _mine_audio_candidates,
    _YTDL_STREAM_OPTS,
    select_search_entry,
)


def _row(fmt: dict[str, Any]) -> str:
    return (
        f"  {str(fmt.get('format_id')):>8}  ext={str(fmt.get('ext')):<5} "
        f"acodec={str(fmt.get('acodec')):<12} abr={str(fmt.get('abr')):<7} "
        f"asr={str(fmt.get('asr')):<6} proto={str(fmt.get('protocol')):<7} "
        f"note={fmt.get('format_note')} lang={fmt.get('language')}"
    )


def render(info: dict[str, Any]) -> list[str]:
    """The whole report, as lines. Pure, so it is testable without the network —
    which is also what keeps the mining call below honest: this passes the same
    info-dict shape the worker slims."""
    formats = info.get("formats") or []
    audio = [f for f in formats if f.get("vcodec") in (None, "none")]
    muxed = [
        f
        for f in formats
        if f.get("vcodec") not in (None, "none")
        and f.get("acodec") not in (None, "none")
        and (f.get("height") or 0) <= 360
    ]
    lines = [
        "SELECTED (what the bot plays)",
        _row(info),
        "",
        f"AUDIO LADDER — {len(audio)} of {len(formats)} formats, worst to best",
        *(_row(f) for f in audio),
        "",
        "MINED CANDIDATES (the retry ladder, best first)",
    ]
    # Mine, then reconstruct — exactly the two steps production takes, so the ladder
    # printed here is the one the retry walks. Mining alone would omit rung 0: the
    # selected format is carried at the top level rather than stored (_candidate_ladder).
    mined = dict(info)
    alternatives = _mine_audio_candidates(info)
    if alternatives:
        mined["audio_candidates"] = alternatives
    candidates = YTDL._candidate_ladder(cast(Any, mined))
    lines.extend(
        f"  {i}. {c.get('format_id')}  {c.get('acodec')}  {c.get('abr')}k"
        for i, c in enumerate(candidates)
    )
    if len(candidates) <= 1:
        # Rung 0 always exists for anything with a stream URL, so what can be missing
        # is the fallbacks — which is the interesting answer for this tool.
        lines.append("  (no alternatives — no format list, or no stream URL)")
    lines += [
        "",
        f"MUXED <=360p fallback rung: {len(muxed)}",
        *(_row(f) for f in muxed),
    ]
    return lines


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python -m src.ytdl_formats <url-or-search>", file=sys.stderr)
        return 2
    # Copy and drop the logger: yt-dlp's warnings are the interesting part here, so
    # they go to stderr rather than through the bot's structlog routing.
    opts = copy.copy(_YTDL_STREAM_OPTS)
    opts.pop("logger", None)
    # cast(), as everywhere yt-dlp's untyped dicts meet ours: the opts profile is the
    # one src.youtube already hands the extractor, and the result is an info-dict.
    raw = yt_dlp.YoutubeDL(cast(Any, opts)).extract_info(
        sys.argv[1], download=False, process=True
    )
    if raw is None:
        print("extraction returned nothing", file=sys.stderr)
        return 1
    info = cast(dict[str, Any], raw)
    if "entries" in info:  # a search result — report the entry that would be played
        # yt_source's own picker, not a copy of its rule: a second copy would drift and
        # this tool would then answer for a song the bot does not choose.
        chosen = select_search_entry(cast(Any, info["entries"]))
        if chosen is None:
            print("search returned no playable entry", file=sys.stderr)
            return 1
        info = cast(dict[str, Any], chosen)
    print("\n".join(render(info)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
