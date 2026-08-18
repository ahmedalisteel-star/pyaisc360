"""Traceability enforcement.

Two guarantees, both cheap and both easy to lose without a test:

1. **Every citation in the source resolves.** A docstring saying "Eq. F2-99" or
   "p. 16.1-99" looks authoritative and is worthless. This scans every module
   for citations and checks each against the generated index.
2. **The roadmap and the library agree** on which chapters exist and where each
   one is going.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

from pyaisc360.core.citations import known_equations, spec_index

SRC = Path(__file__).resolve().parents[1] / "src" / "pyaisc360"
TOOLS = Path(__file__).resolve().parents[1] / "tools"

#: "Eq. F2-8b", "Eq. A-8-1" -- as written in docstrings.
RE_CITED_EQ = re.compile(r"\bEqs?\.\s+((?:A-\d+|[A-N]\d+)-\d+[a-z]?)")
#: "p. 16.1-47"
RE_CITED_PAGE = re.compile(r"p\.\s*16\.1-(\d+)")
#: "Sect. F2.2", "Sect. B4.1"
RE_CITED_SECTION = re.compile(r"\bSect\.\s+([A-N]\d+)")


def source_files() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def _citations(pattern: re.Pattern[str]) -> list[tuple[Path, str]]:
    found: list[tuple[Path, str]] = []
    for path in source_files():
        text = path.read_text(encoding="utf-8")
        found.extend((path, m.group(1)) for m in pattern.finditer(text))
    return found


class TestCitedEquationsExist:
    def test_at_least_one_citation_is_present(self) -> None:
        """Guards against the scan silently matching nothing."""
        assert len(_citations(RE_CITED_EQ)) >= 10

    def test_every_cited_equation_is_in_the_specification(self) -> None:
        labels = known_equations()
        bad = [
            f"{path.relative_to(SRC)}: Eq. {label}"
            for path, label in _citations(RE_CITED_EQ)
            if label not in labels
        ]
        assert not bad, "citations that do not exist in AISC 360-16:\n  " + "\n  ".join(bad)

    def test_every_cited_page_is_within_the_specification(self) -> None:
        pages = {e["page"] for e in spec_index()["equations"] if e["page"]}
        highest = max(pages)
        bad = [
            f"{path.relative_to(SRC)}: p. 16.1-{page}"
            for path, page in _citations(RE_CITED_PAGE)
            if not 1 <= int(page) <= highest + 60  # +60 covers Chapters L-N and appendices
        ]
        assert not bad, "page citations outside the Specification:\n  " + "\n  ".join(bad)

    def test_every_cited_section_exists(self) -> None:
        sections = {s["label"] for s in spec_index()["sections"]}
        bad = [
            f"{path.relative_to(SRC)}: Sect. {label}"
            for path, label in _citations(RE_CITED_SECTION)
            if label not in sections
        ]
        assert not bad, "sections that do not exist:\n  " + "\n  ".join(bad)


class TestEquationPageAgreement:
    """A cited equation and the page cited alongside it must match the index."""

    @pytest.mark.parametrize(
        ("equation", "page"),
        [
            ("B3-1", 12),
            ("B3-2", 12),
            ("E3-2", 35),
            ("E3-3", 35),
            ("E3-4", 36),
            ("F1-1", 46),
            ("F2-5", 48),
            ("F2-7", 48),
            ("F2-8b", 48),
        ],
    )
    def test_docstring_pages_match_the_index(self, equation: str, page: int) -> None:
        from pyaisc360.core.citations import cite

        assert cite(equation).page == page


class TestRoadmapAgreement:
    def test_module_map_covers_every_chapter_with_equations(self) -> None:
        sys.path.insert(0, str(TOOLS))
        from gen_roadmap import MODULE_MAP  # type: ignore[import-not-found]

        chapters = set(spec_index()["counts"]["per_chapter"])
        missing = chapters - set(MODULE_MAP)
        assert not missing, f"chapters with equations but no entry in MODULE_MAP: {missing}"

    def test_implemented_modules_are_in_the_map(self) -> None:
        sys.path.insert(0, str(TOOLS))
        from gen_roadmap import MODULE_MAP  # type: ignore[import-not-found]

        planned = {mod for mod, _, _ in MODULE_MAP.values() if mod != "-"}
        on_disk = {p.name for p in SRC.glob("chapter_*.py")} | {
            p.name for p in SRC.glob("appendix_*.py")
        }
        assert on_disk <= planned, f"modules on disk but not in the roadmap: {on_disk - planned}"


class TestWaveZeroIsComplete:
    """Wave 0 is the foundation every chapter builds on; it ships whole or not at all."""

    @pytest.mark.parametrize(
        "module",
        [
            "pyaisc360.core.units",
            "pyaisc360.core.enums",
            "pyaisc360.core.config",
            "pyaisc360.core.result",
            "pyaisc360.core.citations",
            "pyaisc360.core.exceptions",
            "pyaisc360.utils",
            "pyaisc360.materials",
            "pyaisc360.sections.base",
        ],
    )
    def test_module_imports(self, module: str) -> None:
        __import__(module)

    def test_public_api_is_importable(self) -> None:
        import pyaisc360

        for name in pyaisc360.__all__:
            assert hasattr(pyaisc360, name), name

    def test_library_has_no_runtime_dependencies(self) -> None:
        """It must import inside FreeCAD's embedded interpreter, which has no pip.

        Parsed with :mod:`ast` rather than pattern-matched. The regex this
        replaced flagged a docstring line beginning "from Chapter F may be
        raised..." as a third-party import -- prose and code are not
        distinguishable by pattern, only by parsing.
        """
        stdlib = set(sys.stdlib_module_names)
        offenders: list[str] = []
        for path in source_files():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    roots = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    # node.level > 0 is a relative import: always internal.
                    roots = [] if node.level else [(node.module or "").split(".")[0]]
                else:
                    continue
                offenders.extend(
                    f"{path.relative_to(SRC)}:{node.lineno}: {root}"
                    for root in roots
                    if root and root not in stdlib and root != "pyaisc360"
                )
        assert not offenders, "third-party imports found:\n  " + "\n  ".join(offenders)
