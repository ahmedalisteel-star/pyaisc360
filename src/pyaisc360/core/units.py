"""Unit vocabulary and the SI boundary.

The Specification is written in US customary units (kip, inch, ksi) with metric
equivalents in parentheses, and its numeric coefficients -- 4.71, 1.76, 1.95,
0.658 -- are unit-bearing.  pyaisc360 therefore computes **exclusively in US
customary units** and converts only at the API boundary.  Mixing systems inside
a limit-state equation is the single most common way to get a wrong answer that
still looks plausible, so the conversions live here and nowhere else.

Annotated aliases carry the unit as type metadata.  They are ordinary floats at
runtime -- zero overhead, no wrapper objects inside hot loops -- but a reader
(and mypy, and any tooling that walks ``typing.get_type_hints``) can see that
``Ksi`` and ``Kip`` are not interchangeable.

    def elastic_buckling_stress(slenderness: Ratio, E: Ksi) -> Ksi: ...

AISC 360-16, Section B1 and the Symbols list, pp. 16.1-xxxiii to 16.1-lii.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Final

__all__ = [
    "Unit",
    "Kip",
    "Ksi",
    "Inch",
    "Inch2",
    "Inch3",
    "Inch4",
    "Inch6",
    "KipIn",
    "Ratio",
    "Degrees",
    "Dimensionless",
    "MM_PER_IN",
    "KN_PER_KIP",
    "MPA_PER_KSI",
    "KNM_PER_KIPIN",
    "inch_from_mm",
    "mm_from_inch",
    "in2_from_mm2",
    "mm2_from_in2",
    "in3_from_mm3",
    "mm3_from_in3",
    "in4_from_mm4",
    "mm4_from_in4",
    "in6_from_mm6",
    "mm6_from_in6",
    "kip_from_kn",
    "kn_from_kip",
    "ksi_from_mpa",
    "mpa_from_ksi",
    "kipin_from_knm",
    "knm_from_kipin",
]


@dataclass(frozen=True, slots=True)
class Unit:
    """Type metadata marking the physical unit an annotated float carries."""

    symbol: str
    quantity: str

    def __str__(self) -> str:  # pragma: no cover - display only
        return self.symbol


# --- US customary aliases used throughout the library ---------------------
Kip = Annotated[float, Unit("kip", "force")]
Ksi = Annotated[float, Unit("ksi", "stress")]
Inch = Annotated[float, Unit("in.", "length")]
Inch2 = Annotated[float, Unit("in.^2", "area")]
Inch3 = Annotated[float, Unit("in.^3", "section modulus")]
Inch4 = Annotated[float, Unit("in.^4", "moment of inertia")]
Inch6 = Annotated[float, Unit("in.^6", "warping constant")]
KipIn = Annotated[float, Unit("kip-in.", "moment")]
Ratio = Annotated[float, Unit("-", "slenderness ratio")]
Degrees = Annotated[float, Unit("deg", "angle")]
Dimensionless = Annotated[float, Unit("-", "dimensionless")]


# --- Exact conversion factors ---------------------------------------------
# Exact by definition (international inch, 1959) and by the SI definition of
# the pound-force; the derived factors below are products of these two, so no
# independent rounding creeps in.
MM_PER_IN: Final[float] = 25.4
KN_PER_KIP: Final[float] = 4.4482216152605

MPA_PER_KSI: Final[float] = KN_PER_KIP * 1000.0 / (MM_PER_IN**2)  # ~6.894757
KNM_PER_KIPIN: Final[float] = KN_PER_KIP * MM_PER_IN / 1000.0  # ~0.112985


# --- Length and derived geometric quantities ------------------------------
def inch_from_mm(mm: float) -> Inch:
    """Millimetres to inches. FreeCAD models in mm, so this is the usual entry."""
    return mm / MM_PER_IN


def mm_from_inch(inch: Inch) -> float:
    """Inches to millimetres."""
    return inch * MM_PER_IN


def in2_from_mm2(mm2: float) -> Inch2:
    """Square millimetres to square inches (areas: Ag, An, Ae)."""
    return mm2 / MM_PER_IN**2


def mm2_from_in2(in2: Inch2) -> float:
    return in2 * MM_PER_IN**2


def in3_from_mm3(mm3: float) -> Inch3:
    """Cubic millimetres to cubic inches (section moduli: S, Z)."""
    return mm3 / MM_PER_IN**3


def mm3_from_in3(in3: Inch3) -> float:
    return in3 * MM_PER_IN**3


def in4_from_mm4(mm4: float) -> Inch4:
    """mm^4 to in.^4 (moments of inertia I, torsional constant J)."""
    return mm4 / MM_PER_IN**4


def mm4_from_in4(in4: Inch4) -> float:
    return in4 * MM_PER_IN**4


def in6_from_mm6(mm6: float) -> Inch6:
    """mm^6 to in.^6 (warping constant Cw)."""
    return mm6 / MM_PER_IN**6


def mm6_from_in6(in6: Inch6) -> float:
    return in6 * MM_PER_IN**6


# --- Force, stress, moment -------------------------------------------------
def kip_from_kn(kn: float) -> Kip:
    """Kilonewtons to kips."""
    return kn / KN_PER_KIP


def kn_from_kip(kip: Kip) -> float:
    return kip * KN_PER_KIP


def ksi_from_mpa(mpa: float) -> Ksi:
    """Megapascals to ksi. Fy = 345 MPa -> 50.04 ksi (not the nominal 50)."""
    return mpa / MPA_PER_KSI


def mpa_from_ksi(ksi: Ksi) -> float:
    return ksi * MPA_PER_KSI


def kipin_from_knm(knm: float) -> KipIn:
    """Kilonewton-metres to kip-inches."""
    return knm / KNM_PER_KIPIN


def knm_from_kipin(kipin: KipIn) -> float:
    return kipin * KNM_PER_KIPIN
