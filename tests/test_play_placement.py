"""Tests for src/play_placement.py — the `-play` placement grammar."""

import pytest

from src.play_placement import PlayArgs, PlayMode, split_play_args


class TestSplitPlayArgs:
    """`--now`/`--next` comes off the front of -play's argument, or not at all: the
    remainder is both the search and the origin `-remove` matches on, so a flag
    stripped from mid-line would leave a value the user never typed."""

    @pytest.mark.parametrize(
        "argument,mode,query",
        [
            ("never gonna give you up", PlayMode.NORMAL, "never gonna give you up"),
            ("--now never gonna give you up", PlayMode.NOW, "never gonna give you up"),
            ("--next never gonna give", PlayMode.NEXT, "never gonna give"),
            ("--NOW song", PlayMode.NOW, "song"),
            ("--NeXt song", PlayMode.NEXT, "song"),
            ("--now   https://youtu.be/x", PlayMode.NOW, "https://youtu.be/x"),
            ("--next   https://youtu.be/x", PlayMode.NEXT, "https://youtu.be/x"),
            ("--now", PlayMode.NOW, ""),
            ("--next", PlayMode.NEXT, ""),
            ("  --now  song  ", PlayMode.NOW, "song"),
            ("  --next  song  ", PlayMode.NEXT, "song"),
            # A word that merely starts with a flag is a search, not a flag — and
            # not a typo either, on both the two-dash and one-dash side.
            ("--nowhere man", PlayMode.NORMAL, "--nowhere man"),
            ("-nowhere man", PlayMode.NORMAL, "-nowhere man"),
            ("--nextdoor", PlayMode.NORMAL, "--nextdoor"),
            ("-nextdoor", PlayMode.NORMAL, "-nextdoor"),
            # Trailing and repeated flags stay in the text: only the head is read.
            ("song --now", PlayMode.NORMAL, "song --now"),
            ("song --next", PlayMode.NORMAL, "song --next"),
            ("--now --now song", PlayMode.NOW, "--now song"),
            # The two are mutually exclusive by construction, so the second is
            # search text like any other repeat — it does not combine, and it does
            # not override.
            ("--now --next song", PlayMode.NOW, "--next song"),
            ("--next --now song", PlayMode.NEXT, "--now song"),
            ("", PlayMode.NORMAL, ""),
            ("   ", PlayMode.NORMAL, ""),
        ],
    )
    def test_the_head_decides(self, argument: str, mode: PlayMode, query: str) -> None:
        args = split_play_args(argument)
        assert (args.mode, args.query) == (mode, query)
        assert args.dash_typo is None

    @pytest.mark.parametrize(
        "argument,meant",
        [
            ("-now song", "--now"),  # one ASCII hyphen
            ("–now song", "--now"),  # en dash
            ("—now song", "--now"),  # em dash — what iOS turns a typed `--` into
            ("―now song", "--now"),  # horizontal bar
            ("-–now song", "--now"),  # mixed pair
            ("—NOW song", "--now"),  # the typo is case-insensitive too
            ("-now", "--now"),  # nothing behind it
            ("-next song", "--next"),
            ("—next song", "--next"),
            ("–NEXT song", "--next"),
            ("-–next song", "--next"),
            ("-next", "--next"),
        ],
    )
    def test_a_dash_away_from_a_flag_asks(self, argument: str, meant: str) -> None:
        """These cannot be anything but a misspelt flag, so the command asks rather
        than searching YouTube for the user's own flag — and it names the one it
        thinks was meant, which is the only reason dash_typo carries a string."""
        args = split_play_args(argument)
        assert args.dash_typo == meant
        assert args.mode is PlayMode.NORMAL

    @pytest.mark.parametrize("argument", ["--now song", "--next song"])
    def test_a_real_flag_is_never_read_as_a_typo(self, argument: str) -> None:
        """Ordering inside split_play_args is load-bearing: `--now` satisfies the
        near-miss pattern too (two dashes is within `{1,2}`), so the exact-match
        lookup has to run first or every correct invocation would be answered with
        a did-you-mean."""
        args = split_play_args(argument)
        assert args.dash_typo is None
        assert args.mode is not PlayMode.NORMAL

    @pytest.mark.parametrize(
        "argument",
        [
            "now thats what i call music",
            "now",
            "nowhere",
            "now --now",
            "next to me",
            "next",
            "nextdoor",
        ],
    )
    def test_a_bare_flag_word_is_a_search(self, argument: str) -> None:
        """The did-you-mean deliberately stops at the dash. `-p now thats what i
        call music` and `-p next to me` are real searches, and guessing there would
        break them."""
        args = split_play_args(argument)
        assert (args.mode, args.dash_typo, args.query) == (
            PlayMode.NORMAL,
            None,
            argument,
        )

    def test_the_query_keeps_its_case(self) -> None:
        """Only the head is lowercased to match the flag — the search is what the
        user typed, since it is also the origin -remove matches on."""
        assert split_play_args("--now Never Gonna GIVE").query == "Never Gonna GIVE"

    def test_it_splits_on_any_whitespace(self) -> None:
        """Discord messages carry newlines; the head is a token, not everything up
        to the first space."""
        assert split_play_args("--next\nsong") == PlayArgs(
            mode=PlayMode.NEXT, query="song"
        )

    def test_play_args_is_immutable(self) -> None:
        """Frozen: the split happens once at the top of the body and every consumer
        downstream — the gate, the branch, the origin — reads that same value."""
        args = split_play_args("--now song")
        with pytest.raises(AttributeError):
            setattr(args, "mode", PlayMode.NORMAL)
