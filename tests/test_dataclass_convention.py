"""The dataclass convention over src/, pinned.

CLAUDE.md's code conventions make schema and value objects
`frozen=True, slots=True, kw_only=True`. A decorator that stops one option short
type-checks and lints clean, and every call site is keyword-spelled either way,
so this walk is the only thing that notices."""

import ast
from pathlib import Path

import pytest

import src

_CONVENTION = ("frozen", "slots", "kw_only")

# The options each deliberate exception omits. A reason per entry; an entry that
# no longer matches the class fails test_every_exception_still_needs_its_entry.
_EXCEPTIONS: dict[str, frozenset[str]] = {
    # A mutable request object: place() sets `placed` and a drop sets
    # `dropped_by` after construction, and the registry holds it by identity.
    "src/play_placement.py:PlayRequest": frozenset({"frozen", "kw_only"}),
    # The queued-collection card's accumulator: update() writes `done` and
    # `total` in place while the card's driver reads them.
    "src/queue_progress.py:EnqueueProgress": frozenset({"frozen"}),
}


def _is_dataclass_decorator(node: ast.expr) -> bool:
    """`@dataclass` and `@dataclasses.dataclass`, called or bare."""
    target = node.func if isinstance(node, ast.Call) else node
    if isinstance(target, ast.Name):
        return target.id == "dataclass"
    return isinstance(target, ast.Attribute) and target.attr == "dataclass"


def _missing_options(node: ast.expr) -> frozenset[str]:
    """Which of the three the decorator does not set to a literal True."""
    keywords = node.keywords if isinstance(node, ast.Call) else []
    carried = {
        kw.arg
        for kw in keywords
        if kw.arg is not None
        and isinstance(kw.value, ast.Constant)
        and kw.value.value is True
    }
    return frozenset(name for name in _CONVENTION if name not in carried)


def _scan(tree: ast.Module, where: str) -> dict[str, frozenset[str]]:
    """{"<where>:<Class>": the options it omits} for every dataclass in a tree."""
    found: dict[str, frozenset[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for decorator in node.decorator_list:
            if _is_dataclass_decorator(decorator):
                found[f"{where}:{node.name}"] = _missing_options(decorator)
                break
    return found


@pytest.fixture(scope="module")
def src_dataclasses() -> dict[str, frozenset[str]]:
    """Every src/ dataclass parsed once, from the package's own path, not the CWD."""
    root = Path(src.__file__).parent
    found: dict[str, frozenset[str]] = {}
    for path in sorted(root.rglob("*.py")):
        where = path.relative_to(root.parent).as_posix()
        found.update(_scan(ast.parse(path.read_text()), where))
    return found


_EVERY_SHAPE = """\
import dataclasses
from dataclasses import dataclass

@dataclass(frozen=True, slots=True, kw_only=True)
class Conforming: pass

@dataclasses.dataclass(frozen=True, slots=True, kw_only=True)
class QualifiedConforming: pass

@dataclass(frozen=True, slots=True)
class NoKwOnly: pass

@dataclass(frozen=True)
class NoSlotsNoKwOnly: pass

@dataclass
class Bare: pass

@dataclass(frozen=True, slots=True, kw_only=FLAG)
class NotALiteral: pass

@decorate
class NotADataclass: pass

def outer():
    @dataclass(frozen=True, slots=True)
    class Nested: pass
"""


class TestTheWalkerSeesWhatItClaims:
    """A walker that recognized only one decorator spelling, or read `kw_only`
    off anything other than a literal True, would pass the census by finding
    nothing."""

    def test_it_reports_every_shape_and_nothing_else(self) -> None:
        assert _scan(ast.parse(_EVERY_SHAPE), "x") == {
            "x:Conforming": frozenset(),
            "x:QualifiedConforming": frozenset(),
            "x:NoKwOnly": frozenset({"kw_only"}),
            "x:NoSlotsNoKwOnly": frozenset({"slots", "kw_only"}),
            "x:Bare": frozenset(_CONVENTION),
            "x:NotALiteral": frozenset({"kw_only"}),
            "x:Nested": frozenset({"kw_only"}),
        }


class TestSrcMeetsTheDataclassConvention:
    """Every dataclass in src/ is frozen, slotted and keyword-only, or is named
    here with the options it omits and why."""

    def test_no_class_drops_an_option_unnamed(
        self, src_dataclasses: dict[str, frozenset[str]]
    ) -> None:
        offenders = {
            name: sorted(missing - _EXCEPTIONS.get(name, frozenset()))
            for name, missing in src_dataclasses.items()
            if missing - _EXCEPTIONS.get(name, frozenset())
        }
        assert not offenders

    def test_every_exception_still_needs_its_entry(
        self, src_dataclasses: dict[str, frozenset[str]]
    ) -> None:
        """A class that gains an option, or moves, leaves a stale entry behind
        that would silently exempt whatever takes its name."""
        assert {name: src_dataclasses.get(name) for name in _EXCEPTIONS} == _EXCEPTIONS

    def test_the_walk_reaches_the_whole_package(
        self, src_dataclasses: dict[str, frozenset[str]]
    ) -> None:
        """A walk that found no files, or skipped src/commands/, would pass the
        census above by construction."""
        assert len(src_dataclasses) > 60
        assert any(name.startswith("src/commands/") for name in src_dataclasses)
