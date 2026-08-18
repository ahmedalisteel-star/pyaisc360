"""Traceability: every computed number knows which equation produced it.

Citations are not typed by hand.  ``tools/scan_spec.py`` indexes the source text
once into ``data/spec_index.json``; :func:`cite` looks a label up in that index
and returns the section and *Specification* page it was found on.  An equation
label that is not in the index raises, so a typo in a docstring citation becomes
a test failure rather than a plausible-looking wrong reference.

    >>> cite("F2-2").text
    'AISC 360-16, Eq. F2-2 (Sect. F2), p. 16.1-47'
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .exceptions import AISC360Error

__all__ = ["Citation", "cite", "spec_index", "known_equations"]

SPEC_NAME = "AISC 360-16"
_INDEX_PATH = Path(__file__).resolve().parent.parent / "data" / "spec_index.json"


class UnknownEquationError(AISC360Error, KeyError):
    """The cited equation label does not exist in the indexed Specification."""


@dataclass(frozen=True, slots=True)
class Citation:
    """A pointer back into ANSI/AISC 360-16.

    Attributes
    ----------
    equation:
        Equation label as printed, e.g. ``"F2-8b"`` or ``"A-8-1"``.
    section:
        Owning section, e.g. ``"F2"``.
    page:
        *Specification* page, i.e. the ``nn`` in ``16.1-nn``.
    line:
        1-based line in the indexed source text -- for auditing the transcription.
    """

    equation: str
    section: str
    page: int | None = None
    line: int | None = None

    @property
    def text(self) -> str:
        """Single-line citation suitable for a calculation sheet."""
        page = f", p. 16.1-{self.page}" if self.page is not None else ""
        return f"{SPEC_NAME}, Eq. {self.equation} (Sect. {self.section}){page}"

    def __str__(self) -> str:
        return self.text


@lru_cache(maxsize=1)
def spec_index() -> dict[str, Any]:
    """Load and cache the generated Specification index."""
    if not _INDEX_PATH.exists():  # pragma: no cover - packaging failure
        raise AISC360Error(
            f"Specification index missing at {_INDEX_PATH}. "
            "Regenerate it with: py tools/scan_spec.py --source <A360-16W.txt> "
            "--out src/pyaisc360/data/"
        )
    index: dict[str, Any] = json.loads(_INDEX_PATH.read_text(encoding="utf-8"))
    return index


@lru_cache(maxsize=1)
def _by_label() -> dict[str, Citation]:
    return {
        e["label"]: Citation(
            equation=e["label"], section=e["section"], page=e["page"], line=e["line"]
        )
        for e in spec_index()["equations"]
    }


def cite(equation: str) -> Citation:
    """Resolve an equation label to its :class:`Citation`.

    Raises
    ------
    UnknownEquationError
        If the label is not in the indexed Specification -- which means either a
        typo, or a "derived" relationship that should be documented in prose
        rather than cited as an equation.
    """
    try:
        return _by_label()[equation]
    except KeyError:
        raise UnknownEquationError(
            f"{equation!r} is not a numbered equation in {SPEC_NAME}. "
            "Check the label, or document the step in prose instead of citing it."
        ) from None


def known_equations() -> frozenset[str]:
    """Every equation label in the Specification. Used by ``tests/test_coverage.py``."""
    return frozenset(_by_label())
