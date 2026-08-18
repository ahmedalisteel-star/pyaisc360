"""Step 1 of the pyaisc360 pipeline: scan the AISC 360-16 source text and emit a
machine-readable index of every equation, table and section.

The source text is a PDF text dump, so it is *not* clean enough to auto-generate
code from: subscripts land on their own lines (``P =F A`` / ``n  cr g``) and
Greek letters are frequently mangled.  What the dump *is* reliable for is
locating things.  This scanner therefore produces a **traceability index**
(equation label -> line, spec page, owning section) which the implementation
modules cite in their docstrings and which the test suite uses to assert that
every catalogued equation has an implementation or an explicit waiver.

Usage:
    py tools/scan_spec.py --source "<path to A360-16W.txt>" --out data/
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------
# Structure of the source dump.  Located by scanning for the chapter banners;
# the document repeats them a second time for the (non-mandatory) Commentary,
# so we key off the *first* occurrence of each and stop at the Commentary.
# --------------------------------------------------------------------------

CHAPTERS = "ABCDEFGHIJKLMN"

RE_CHAPTER = re.compile(r"^CHAPTER ([A-N])\s*$")
RE_APPENDIX = re.compile(r"^APPENDIX (\d)\s*$")
RE_COMMENTARY = re.compile(r"^COMMENTARY\s*$")
RE_PAGE = re.compile(r"16\.1-(\d+)")
# Equation labels: (E3-1), (F2-8b), (A-6-1), (I3-1a) ...
RE_EQ = re.compile(r"\((?:([A-N])(\d+)|A-(\d+))-(\d+)([a-z]?)\)")
RE_TABLE = re.compile(r"^TABLE ((?:[A-N]|A-)[\w.\-]+)(?:\s+\(continued\))?\s*$")
# Section headings, e.g. "F2. DOUBLY SYMMETRIC COMPACT ..."  Headings are all
# caps in the body; the chapter's own contents list uses title case, which we
# reject by requiring at least two consecutive capitals in the title.
RE_SECTION = re.compile(r"^([A-N]\d+)\.\s+([A-Z][A-Z ,\-/]{4,})")
RE_APP_SECTION = re.compile(r"^(\d+\.\d+)\.\s+([A-Z][A-Z ,\-/]{4,})")


#: Known defects in the PDF text extraction, corrected explicitly rather than
#: silently. Each entry records the line, what the dump says, what the printed
#: Specification actually says, and how that was established.
#:
#: The extractor interleaves a two-column table cell character by character and
#: can emit the wrong label. Every correction here has been read back out of the
#: interleaved fragment and cross-checked against the equation it labels.
EXTRACTION_CORRECTIONS: list[dict[str, object]] = [
    {
        "line": 10709,
        "dump_says": "K2-2a",
        "actually": "K2-2b",
        "section": "K2",
        "why": (
            "Table K2.1's longitudinal-plate row prints (K2-2a) for the axial "
            "equation and (K2-2b) for Mn = 0.8*lb*Rn. The dump emits (K2-2a) "
            "twice; the label (K2-2b) is legible interleaved down the mangled "
            "cell as '(K' '2' '-2' 'b' ')'. Table K2.1's transverse-plate row "
            "above it shows the same two-label pattern extracted correctly as "
            "K2-1b/K2-1a, confirming the layout."
        ),
    },
]


@dataclass
class Equation:
    label: str
    chapter: str
    section: str
    line: int
    page: int | None
    context: str


@dataclass
class Table:
    label: str
    chapter: str
    line: int
    page: int | None


@dataclass
class Section:
    label: str
    title: str
    chapter: str
    line: int
    page: int | None
    equations: list[str] = field(default_factory=list)


def _clean(text: str) -> str:
    """Strip the recurring PDF furniture from a context line."""
    text = text.replace("(cid:129)", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def find_body(lines: list[str]) -> tuple[int, int]:
    """Return (start, end) line indices of the mandatory Specification body.

    The body runs from ``CHAPTER A`` to the start of the Commentary, which the
    dump introduces by repeating ``CHAPTER A`` a second time.
    """
    chapter_a = [i for i, ln in enumerate(lines) if RE_CHAPTER.match(ln) and ln.strip().endswith("A")]
    if len(chapter_a) < 2:
        raise ValueError("could not locate the Specification/Commentary split")
    return chapter_a[0], chapter_a[1]


def scan(path: Path) -> dict:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    start, end = find_body(lines)

    equations: list[Equation] = []
    tables: list[Table] = []
    sections: list[Section] = []

    page: int | None = None
    chapter = "A"
    section = "A1"
    by_section: dict[str, Section] = {}
    seen_eq: set[str] = set()

    for idx in range(start, end):
        raw = lines[idx]
        lineno = idx + 1  # 1-based, matches an editor

        m_page = RE_PAGE.search(raw)
        if m_page:
            page = int(m_page.group(1))

        m_ch = RE_CHAPTER.match(raw)
        if m_ch:
            chapter = m_ch.group(1)
            continue

        m_app = RE_APPENDIX.match(raw)
        if m_app:
            chapter = f"A{m_app.group(1)}"
            continue

        m_sec = RE_SECTION.match(raw) or RE_APP_SECTION.match(raw)
        if m_sec:
            label, title = m_sec.group(1), _clean(m_sec.group(2))
            # Long headings wrap; the continuation lines are all-caps too.
            for cont in lines[idx + 1 : idx + 4]:
                stripped = _clean(cont)
                if re.fullmatch(r"[A-Z][A-Z ,\-/]{3,}", stripped):
                    title += " " + stripped
                else:
                    break
            title = title.title()
            section = label
            if label not in by_section:
                sec = Section(label=label, title=title, chapter=chapter, line=lineno, page=page)
                by_section[label] = sec
                sections.append(sec)
            continue

        m_tab = RE_TABLE.match(raw)
        if m_tab and "continued" not in raw:
            tables.append(Table(label=m_tab.group(1), chapter=chapter, line=lineno, page=page))

        correction = next(
            (c for c in EXTRACTION_CORRECTIONS if c["line"] == lineno), None
        )
        if correction is not None:
            label = str(correction["actually"])
            if label not in seen_eq:
                seen_eq.add(label)
                equations.append(
                    Equation(
                        label=label,
                        chapter=label.split("-")[0][0],
                        section=str(correction["section"]),
                        line=lineno,
                        page=page,
                        context=f"CORRECTED from {correction['dump_says']}: {correction['why']}",
                    )
                )
                if section in by_section:
                    by_section[section].equations.append(label)
            continue

        for m_eq in RE_EQ.finditer(raw):
            if m_eq.group(1):  # chapter equation, e.g. F2-8b
                label = f"{m_eq.group(1)}{m_eq.group(2)}-{m_eq.group(4)}{m_eq.group(5)}"
                eq_chapter = m_eq.group(1)
                eq_section = f"{m_eq.group(1)}{m_eq.group(2)}"
            else:  # appendix equation, e.g. A-6-1
                label = f"A-{m_eq.group(3)}-{m_eq.group(4)}{m_eq.group(5)}"
                eq_chapter = f"A{m_eq.group(3)}"
                eq_section = eq_chapter
            if label in seen_eq:
                continue
            seen_eq.add(label)
            equations.append(
                Equation(
                    label=label,
                    chapter=eq_chapter,
                    section=eq_section,
                    line=lineno,
                    page=page,
                    context=_clean(raw)[:160],
                )
            )
            if section in by_section:
                by_section[section].equations.append(label)

    per_chapter = Counter(e.chapter for e in equations)
    per_section: dict[str, int] = defaultdict(int)
    for e in equations:
        per_section[e.section] += 1

    return {
        "source": path.name,
        "spec": "ANSI/AISC 360-16, July 7, 2016",
        "body_lines": [start + 1, end],
        "corrections": EXTRACTION_CORRECTIONS,
        "counts": {
            "equations": len(equations),
            "tables": len(tables),
            "sections": len(sections),
            "per_chapter": dict(sorted(per_chapter.items())),
            "per_section": dict(sorted(per_section.items())),
        },
        "equations": [asdict(e) for e in equations],
        "tables": [asdict(t) for t in tables],
        "sections": [asdict(s) for s in sections],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    index = scan(args.source)
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / "spec_index.json"
    target.write_text(json.dumps(index, indent=1), encoding="utf-8")

    c = index["counts"]
    print(f"wrote {target}")
    print(f"  equations : {c['equations']}")
    print(f"  tables    : {c['tables']}")
    print(f"  sections  : {c['sections']}")
    for ch, n in c["per_chapter"].items():
        print(f"    {ch:>3}: {n}")


if __name__ == "__main__":
    main()
