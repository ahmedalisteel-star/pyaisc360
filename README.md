# pyaisc360

Executable design verification to **ANSI/AISC 360-16**, *Specification for Structural
Steel Buildings* (July 7, 2016) — organised chapter-by-chapter, typed, and traceable
back to the equation number and page it came from.

Built to drop into open-source structural tooling (FreeCAD / BIMSteelAuto, OpenSees,
BlenderBIM). No runtime dependencies, so it imports inside FreeCAD's embedded
interpreter.

## Install

```bash
pip install pyaisc360
```

Requires Python 3.10+. No runtime dependencies.

## Bring your own sections

There is no shapes database here — the AISC *Shapes Database* is a separate
publication, and most users already have a section library. Point the adapter at
whatever you have, in whatever units and under whatever names:

```python
from pyaisc360 import SectionAdapter

# a metric profile record with its own naming
ipe300 = SectionAdapter(profile, metric=True, overrides={"Ag": "A_mm2", "Zx": "Wpl_y"})
ipe300.Ag      # in.^2, converted
ipe300.rts     # derived from Iy, Cw and Sx per Eq. F2-7 when not tabulated
```

## Every result shows its workings

```
==============================================================================
Mn -- LRFD (phi = 0.90)
==============================================================================
limit state                                         Mn [kip-in.]
------------------------------------------------------------------------------
yielding (plastic moment)                                  5,050
    Mp = Fy*Zx = 50 x 101
lateral-torsional buckling                                 4,550   <-- governs
    Lp < Lb <= Lr, inelastic LTB
------------------------------------------------------------------------------
governing: AISC 360-16, Eq. F2-2 (Sect. F2), p. 16.1-47
Mn                                                         4,550
phi*Mn                                                     4,095
------------------------------------------------------------------------------
Ru                                                         3,000
demand / capacity                                          0.733   OK
==============================================================================
```

## Status

**Step 1 — scan and metadata mapping: complete.** 372 equations, 43 tables and 120
sections catalogued in [`spec_index.json`](src/pyaisc360/data/spec_index.json), with
the build plan in [`ROADMAP.md`](ROADMAP.md).

**Step 2 — Wave 0 foundations: complete.** `core/` (units, config, results,
citations, exceptions), `utils.py`, `materials.py` and `sections/`. 175 tests, ruff
clean, mypy strict clean.

**Wave 1 — Chapter E: complete.** All 27 equations of Sects. E3–E7, plus Table B4.1a
in `chapter_b.py`. The Eq. E4-4 cubic is solved in closed form. 312 tests, including
AISC *Manual* Table 4-1a benchmarks and a regression lock confirmed bit-for-bit
against an independent Chapter E implementation over 184 shape/length combinations.

**Wave 1 - Chapter F: complete.** All 108 equations of Sects. F2-F13, plus Table
B4.1b. Validated against AISC *Manual* Table 3-2 (W18x50, W16x26, W24x68, W24x55).
451 tests.

**Wave 2 - Chapters G and D: complete.** All 21 Chapter G equations (Sects. G1-G6,
including the Sect. G2.1(a) phi = 1.00 exception, tension field action and stiffener
sizing) and all 5 Chapter D equations plus the eight cases of Table D3.1. 594 tests.

**Wave 2 - Chapter H and Appendices 6/7/8: complete.** All 16 Chapter H equations
(interaction, HSS torsion, flange rupture) and all 28 appendix equations (stability
bracing, alternative stability methods, B1/B2 amplification). 722 tests.

**Wave 3 - Chapter J: complete.** All 44 equations of Sects. J2-J10, plus Tables
J3.1 (minimum pretension), J3.2 (nominal fastener stresses) and the J2.5 weld
strengths. 837 tests.

**Wave 5 - Chapter C: complete.** All 3 equations of Sects. C1-C3 -- notional loads,
the tau_b stiffness reduction, and the Sect. C3 effective-length rule -- wired into
`DesignSettings.stability_method` and coupled to the Appendix 8 B1/B2 amplifiers.
912 tests.

**Wave 5 - Chapter K: complete.** All 47 equations of Sects. K1-K5 -- plate-to-HSS,
HSS-to-HSS truss and moment connections, and welds to rectangular HSS. Recovering
Eq. K2-2b from a text-extraction defect raised the Specification's catalogued total
from 372 to 373. 996 tests.

**Wave 6 - Chapter I and Appendices 1-4: complete.** All 39 Chapter I equations
(Sects. I1-I8: encased and filled columns, composite flexure, the Sect. I5
interaction pair, load transfer, and stud and channel anchors) and all 33
equations of Appendices 1-4 (inelastic analysis, ponding, fatigue, fire).
1,138 tests.

**373 of 373 equations implemented -- the Specification is complete.** Chapters
B through K plus Appendices 1-4 and 6-8. Chapters A, L, M and N carry no
equations: they are scope, serviceability, fabrication and quality-assurance
provisions rather than design calculations.

```bash
python -m pytest tests/ -q
```

## Design rules

* **Chapter = module.** `chapter_e.py` is Chapter E and nothing else.
* **LRFD and ASD are one code path.** Every public strength function returns nominal
  strength plus the applicable φ and Ω, and a single `Basis` flag selects which is
  applied (Eq. B3-1 / B3-2). No forked implementations.
* **Every function cites its source** in the docstring: equation number, section, and
  *Specification* page (`16.1-nn`), taken from the generated index — not typed by hand.
* **Nothing is silently approximate.** A provision outside a function's stated range of
  applicability raises, it does not extrapolate.
* **Units are explicit.** US customary (kip, in., ksi) internally, with a conversion
  layer at the boundary; unit intent is carried in the type annotations.

## Regenerating the index

```bash
python tools/scan_spec.py --source /path/to/A360-16W.txt --out data/
python tools/gen_roadmap.py
```

## Copyright note

ANSI/AISC 360-16 is copyrighted by the American Institute of Steel Construction. This
repository contains **no reproduction of the Specification text** — only the design
equations it defines (which are not themselves copyrightable) and citations by section,
equation and page number. Users are expected to hold their own copy of the
Specification and of the AISC *Shapes Database*, which is a separate publication and is
therefore loaded at runtime rather than vendored here.

This library is an aid to a qualified engineer, not a substitute for one. Every result
is subject to the same professional review as a hand calculation.

## License

MIT. See [LICENSE](LICENSE).
