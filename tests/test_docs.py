"""Integrity of the tracked documentation's cross-references.

`docs/ARCHITECTURE.md` is the only tracked file under `docs/`, and comments across
`src/` link to its anchors instead of repeating the long-form context. Renaming one of
its headings orphans every one of those pointers silently — nothing else in the build
reads them, so a reader following the link is the first to find out. Same for the
README's command table, which is the only place a user learns a command exists.

Static text checks, deliberately: they cost microseconds, they need no fixtures, and
the thing they protect is agreement between files rather than any runtime behaviour.
"""

import re
from pathlib import Path
from typing import Final

_ROOT: Final = Path(__file__).resolve().parent.parent
_ARCHITECTURE: Final = _ROOT / "docs" / "ARCHITECTURE.md"

# Files whose ARCHITECTURE.md pointers are checked: everything under src/ and tests/,
# plus CLAUDE.md, which cites four anchors of its own and drifts just as quietly.
_CITING_GLOBS: Final = ("src/**/*.py", "tests/**/*.py")

_HEADING: Final = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)
_CITATION: Final = re.compile(r"ARCHITECTURE\.md#([\w-]+)")
# A markdown link into this same file: the table of contents and the in-body
# cross-references, which orphan on a rename exactly like an external citation.
_SELF_LINK: Final = re.compile(r"\]\(#([\w-]+)\)")

# The first column of any pipe-table row whose cell is a `-command` code span.
_README_COMMAND: Final = re.compile(r"^\|\s*`-([a-z]+)[^`]*`\s*\|", re.MULTILINE)
_COG_COMMAND: Final = re.compile(r"@commands\.command\(\s*name=\"([a-z]+)\"")


def _slug(heading: str) -> str:
    """GitHub's heading-to-anchor rule, which three details make non-obvious:
    underscores SURVIVE (`why-query_source-...`), a leading hyphen survives
    (`` `-play` `` → `-play-...`), and each space becomes its own hyphen rather than
    runs collapsing — so `Pause / Resume` is `pause--resume`, with two."""
    return re.sub(r"\s", "-", re.sub(r"[^\w\s-]", "", heading.strip().lower()))


def _anchors() -> set[str]:
    return {_slug(m.group(1)) for m in _HEADING.finditer(_ARCHITECTURE.read_text())}


def _citing_files() -> list[Path]:
    files = [p for glob in _CITING_GLOBS for p in _ROOT.glob(glob)]
    files.append(_ROOT / "CLAUDE.md")
    return files


class TestArchitectureAnchors:
    """CLAUDE.md rule 2 states the hazard and that "nothing checks them yet"."""

    def test_every_cited_anchor_resolves(self) -> None:
        anchors = _anchors()
        orphaned = {
            f"{path.relative_to(_ROOT)} -> #{m.group(1)}"
            for path in _citing_files()
            for m in _CITATION.finditer(path.read_text())
            if m.group(1) not in anchors
        }
        assert not orphaned, (
            "docs/ARCHITECTURE.md heading renamed or removed, orphaning a pointer:\n  "
            + "\n  ".join(sorted(orphaned))
        )

    def test_every_self_link_resolves(self) -> None:
        """The table of contents is the densest set of these pointers, and the one a
        heading rename breaks first."""
        text = _ARCHITECTURE.read_text()
        anchors = _anchors()
        orphaned = {m.group(1) for m in _SELF_LINK.finditer(text)} - anchors
        assert not orphaned, (
            "ARCHITECTURE.md links to its own missing headings: "
            + ", ".join(f"#{a}" for a in sorted(orphaned))
        )

    def test_no_duplicate_headings(self) -> None:
        """GitHub disambiguates a repeated heading by suffixing `-1`, `-2`, so the
        second one's anchor is not what `_slug` computes and every check above would
        silently stop covering it."""
        slugs = [
            _slug(m.group(1)) for m in _HEADING.finditer(_ARCHITECTURE.read_text())
        ]
        duplicated = sorted({s for s in slugs if slugs.count(s) > 1})
        assert not duplicated, f"headings collide on one anchor: {duplicated}"

    def test_the_citations_are_actually_found(self) -> None:
        """A regex that matched nothing would make all of the above vacuously pass."""
        found = sum(len(_CITATION.findall(p.read_text())) for p in _citing_files())
        assert found >= 15, f"only {found} anchor citations found — regex broken?"


class TestReadmeCommandTable:
    """The README's table is where a user learns a command exists; nothing else
    surfaces it. `test_help.py` already pins the in-app help the same way."""

    def test_every_command_is_documented(self) -> None:
        readme = (_ROOT / "README.md").read_text()
        documented = set(_README_COMMAND.findall(readme))
        declared = set(
            _COG_COMMAND.findall((_ROOT / "src" / "musicbot.py").read_text())
        )
        assert declared, "no @commands.command(name=...) found — regex broken?"
        assert not declared - documented, (
            "command missing from the README table: "
            + ", ".join(f"-{c}" for c in sorted(declared - documented))
        )

    def test_the_table_documents_no_command_that_does_not_exist(self) -> None:
        readme = (_ROOT / "README.md").read_text()
        documented = set(_README_COMMAND.findall(readme))
        declared = set(
            _COG_COMMAND.findall((_ROOT / "src" / "musicbot.py").read_text())
        )
        # `-help` is a discord.py HelpCommand, not a cog command, so it is documented
        # without ever appearing in musicbot.py's decorators.
        assert not documented - declared - {"help"}, (
            "README documents a command that no longer exists: "
            + ", ".join(f"-{c}" for c in sorted(documented - declared - {"help"}))
        )
