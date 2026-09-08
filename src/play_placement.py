"""`-play`'s placement grammar: the flag a caller writes, and where the songs go.

`split_play_args` is the parse, `PlayMode` is what it found, and `Placement` is what
the pipeline does about it. A module of its own rather than part of `play_pipeline`,
because the command has to parse its argument BEFORE it may pick a concurrency
bucket — see `src/commands/play.py`.
"""

import re
from dataclasses import dataclass
from enum import Enum
from typing import Final, Optional

NOW_FLAG: Final[str] = "--now"
NEXT_FLAG: Final[str] = "--next"


class PlayMode(Enum):
    """Where a `-play` invocation puts its song.

    One field, so `now and next` is unrepresentable.
    """

    NORMAL = "normal"
    NOW = "now"
    NEXT = "next"


_FLAG_MODES: Final[dict[str, PlayMode]] = {
    NOW_FLAG: PlayMode.NOW,
    NEXT_FLAG: PlayMode.NEXT,
}


class Placement(Enum):
    """Where an enqueue puts its songs, and which confirmation says so. COLD_FRONT
    and NEXT both front-insert, but only a disconnected bot waking a persisted
    queue earns the resume notice."""

    TAIL = "tail"
    COLD_FRONT = "cold_front"
    NEXT = "next"


# Every dash Unicode offers that a keyboard or a paste substitutes for ASCII `-`:
# hyphen, non-breaking hyphen, figure dash, en dash, em dash, horizontal bar. iOS
# turns a typed `--` into a single em dash.
_DASHES: Final[str] = "-‐‑‒–—―"
# Alternation built from _FLAG_MODES' own keys, so a renamed flag cannot leave this
# branch offering one that no longer exists. The group is a flag minus its dashes;
# split_play_args re-attaches them for the reply.
_NEAR_FLAG_RE: Final[re.Pattern[str]] = re.compile(
    f"[{_DASHES}]{{1,2}}({'|'.join(flag[2:] for flag in _FLAG_MODES)})"
)


@dataclass(frozen=True, slots=True, kw_only=True)
class PlayArgs:
    """`-play`'s argument, split into the placement flag and the query. kw_only:
    `query` and `dash_typo` are adjacent strings and one is echoed into an embed.
    `dash_typo` names the flag a misspelt leading token meant; it only accompanies
    PlayMode.NORMAL."""

    mode: PlayMode
    query: str
    dash_typo: Optional[str] = None


def split_play_args(argument: str) -> PlayArgs:
    """Split a leading `--now`/`--next` off `-play`'s argument. Only the FIRST token
    counts, so a flag further along stays part of the search and of the origin
    `-remove` matches on; one flag, never a run. A leading token one dash off a
    flag (`-now`, an autocorrected `—next`) sets `dash_typo`; the exact match runs
    first, since a real `--now` also fits the near-miss pattern. A bare `now`/`next`
    is a search (`-p next to me`)."""
    stripped = argument.strip()
    parts = stripped.split(maxsplit=1)
    if not parts:
        return PlayArgs(mode=PlayMode.NORMAL, query="")
    head = parts[0].lower()
    # No strip on the tail: `stripped` had none, and split() eats the separator run.
    rest = parts[1] if len(parts) > 1 else ""
    mode = _FLAG_MODES.get(head)
    if mode is not None:
        return PlayArgs(mode=mode, query=rest)
    typo = _NEAR_FLAG_RE.fullmatch(head)
    if typo is not None:
        return PlayArgs(
            mode=PlayMode.NORMAL, query=stripped, dash_typo=f"--{typo.group(1)}"
        )
    return PlayArgs(mode=PlayMode.NORMAL, query=stripped)
