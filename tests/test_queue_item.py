"""Tests for src/queue_item.py: the queue item's declaration contracts."""

import ast
import dataclasses
import pathlib
from dataclasses import FrozenInstanceError, replace
from unittest.mock import MagicMock

import pytest

from src.queue_item import NpCard, NpHostRef, QueueObject


class TestQueueObject:
    def test_required_fields(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            webpage_url="https://www.youtube.com/watch?v=abc",
            title="My Song",
            requester=mock_author,
        )
        assert qobj.webpage_url == "https://www.youtube.com/watch?v=abc"
        assert qobj.title == "My Song"
        assert qobj.requester is mock_author

    def test_ts_defaults_to_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Title", requester=mock_author
        )
        assert qobj.ts is None

    def test_ts_can_be_set(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            webpage_url="https://yt.com/watch?v=1",
            title="Title",
            requester=mock_author,
            ts=90,
        )
        assert qobj.ts == 90

    def test_optional_fields_default_to_none(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Title", requester=mock_author
        )
        assert qobj.user_input is None
        assert qobj.duration is None
        assert qobj.uploader is None

    def test_optional_fields_can_be_set(self, mock_author: MagicMock) -> None:
        qobj = QueueObject(
            webpage_url="https://yt.com/watch?v=1",
            title="Title",
            requester=mock_author,
            user_input="search term",
            duration=180,
            uploader="My Channel",
        )
        assert qobj.user_input == "search term"
        assert qobj.duration == 180
        assert qobj.uploader == "My Channel"

    def test_is_dataclass(self, mock_author: MagicMock) -> None:
        assert dataclasses.is_dataclass(QueueObject)

    def test_two_asks_for_one_song_are_two_items(self, mock_author: MagicMock) -> None:
        """An item is one ask: a second ask for the same song, built the same way,
        is a different item, as it is on the deque."""
        q1 = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Song", requester=mock_author
        )
        q2 = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Song", requester=mock_author
        )
        assert q1 == q1
        assert q1 != q2

    def test_a_queued_item_is_frozen(self, mock_author: MagicMock) -> None:
        item = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Song", requester=mock_author
        )
        with pytest.raises(FrozenInstanceError):
            setattr(item, "title", "Retitled")

    def test_an_item_hashes_by_identity(self, mock_author: MagicMock) -> None:
        """Pins the class comment: a set tells two asks for one song apart the way
        the queue does (holds, display_index, _listed), and a resume tail carrying
        a card with a live host ref hashes too, its own_embeds list notwithstanding."""
        first = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Song", requester=mock_author
        )
        second = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Song", requester=mock_author
        )
        assert first is not second
        assert len({first, second}) == 2
        tail = replace(
            first,
            np_card=NpCard(
                message_id=1,
                channel_id=2,
                dedicated=True,
                host_ref=NpHostRef(message=MagicMock(), own_embeds=[]),
            ),
        )
        assert hash(tail) == hash(tail)

    def test_fields_are_named_at_construction(self, mock_author: MagicMock) -> None:
        # webpage_url and title are both str: positional, either order type-checks.
        with pytest.raises(TypeError):
            QueueObject("https://yt.com/watch?v=1", "Song", mock_author)  # pyright: ignore[reportCallIssue]

    def test_every_field_takes_part_in_replace(self) -> None:
        """replace() carries the whole ask across the three rebuilds and the
        crash restore, and it silently skips a field declared init=False — such a
        field would be reset to its default at every rebuild."""
        assert [f.name for f in dataclasses.fields(QueueObject) if not f.init] == []

    def test_asdict_and_vars_stay_off_the_item(self, mock_author: MagicMock) -> None:
        """vars() raises on a slotted item, but asdict() does not: it reads
        fields() and deep-copies every value, including the Member on requester.
        The wire tables spell the fields out instead. Pins the class comment: no
        asdict() call anywhere in src/, qualified or not, and no vars() in a
        module that names QueueObject (argparse's vars(args) elsewhere is fine)."""
        item = QueueObject(
            webpage_url="https://yt.com/watch?v=1", title="Song", requester=mock_author
        )
        assert not hasattr(item, "__dict__")
        src = pathlib.Path(__file__).resolve().parents[1] / "src"
        hits: list[str] = []
        for path in sorted(src.rglob("*.py")):
            text = path.read_text()
            handles_item = "QueueObject" in text
            for node in ast.walk(ast.parse(text)):
                if not isinstance(node, ast.Call):
                    continue
                called = getattr(node.func, "id", None) or getattr(
                    node.func, "attr", None
                )
                if called == "asdict" or (called == "vars" and handles_item):
                    hits.append(f"{path.relative_to(src.parent)}:{node.lineno}")
        assert hits == []
