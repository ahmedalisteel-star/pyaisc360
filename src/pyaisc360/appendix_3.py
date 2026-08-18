"""Appendix 3 -- Fatigue.

Covers Sects. 3.1 through 3.5, pp. 16.1-196 to 16.1-221.

Fatigue is the one place in the Specification where **stress range** governs
rather than peak stress, and where the answer depends on how many cycles the
detail will see rather than on any strength. Two consequences shape this module:

* there is **no phi and no Omega**. Eq. A-3-1 produces an allowable stress
  range directly, and Sect. 3.1 states the provisions apply "to stresses
  calculated on the basis of service loads" -- not factored ones. Feeding LRFD
  stresses in overstates the demand by the load factors.
* every detail has a **threshold** ``FTH`` below which fatigue need not be
  considered at all, regardless of cycle count. Eq. A-3-1's ``>= FTH`` is that
  floor, and it is what makes indefinite design life possible.

The stress categories and their ``Cf``/``FTH`` pairs live in Table A-3.1, which
runs to twenty pages of illustrated details. The eight standard categories are
tabulated here; assigning a detail to one of them is an engineering judgement
this module cannot make.
"""

from __future__ import annotations

import math
from enum import Enum

from .core.exceptions import AISC360Error, GeometryError
from .core.units import Inch, Inch2, Ksi

__all__ = [
    "StressCategory",
    "TABLE_A3_1",
    "allowable_stress_range",
    "category_F_stress_range",
    "PJP_reduction_factor",
    "fillet_reduction_factor",
    "PJP_root_stress_range",
    "fillet_root_stress_range",
    "bolt_tensile_area",
    "fatigue_need_not_be_considered",
]


class StressCategory(str, Enum):
    """Fatigue stress categories of Table A-3.1, pp. 16.1-202 to 16.1-221."""

    A = "A"
    B = "B"
    B_PRIME = "B'"
    C = "C"
    C_PRIME = "C'"
    C_DOUBLE_PRIME = "C''"
    D = "D"
    E = "E"
    E_PRIME = "E'"
    F = "F"


#: Table A-3.1 -- ``(Cf, FTH)``, pp. 16.1-202 to 16.1-215. ``Cf`` is the fatigue
#: constant and ``FTH`` the threshold allowable stress range in ksi for
#: indefinite design life.
#:
#: **These are the 360-16 constants, and they are NOT the 360-10 ones.** The
#: 2010 Specification tabulated ``Cf`` as 250x10^8 for category A and wrote
#: Eq. A-3-1 without a leading coefficient; 360-16 moved a factor of 10^9 out
#: of the table and into the equation, which now reads ``1,000*(Cf/nSR)^0.333``
#: with ``Cf = 25``. The two forms give the same answer only if the table and
#: the equation come from the same edition -- mixing them is off by 10^9 in
#: either direction, and in the unconservative direction it returns stress
#: ranges near 29,000 ksi, so every fatigue check passes and nothing looks
#: wrong until someone reads the number.
#:
#: Categories C' and C'' have no tabulated pair: Sects. 3.3(c) and 3.3(d)
#: compute their stress ranges directly from Eqs. A-3-3 and A-3-5 with a
#: reduction factor, rather than from Eq. A-3-1.
TABLE_A3_1: dict[StressCategory, tuple[float, Ksi]] = {
    StressCategory.A: (25.0, 24.0),
    StressCategory.B: (12.0, 16.0),
    StressCategory.B_PRIME: (6.1, 12.0),
    StressCategory.C: (4.4, 10.0),
    StressCategory.D: (2.2, 7.0),
    StressCategory.E: (1.1, 4.5),
    StressCategory.E_PRIME: (0.39, 2.6),
}


def allowable_stress_range(category: StressCategory, nSR: float) -> tuple[Ksi, str]:
    """Allowable stress range for categories A through E'.

    AISC 360-16, Eq. A-3-1, Sect. 3.3(a), p. 16.1-197::

        FSR = 1000*(Cf/nSR)^0.333  >=  FTH

    ``nSR`` is the number of stress-range fluctuations in the design life -- not
    a rate. A detail at 500 cycles a day over 50 years sees about 9.1 million.

    The ``>= FTH`` floor is what makes indefinite design life possible: past the
    cycle count at which the curve drops below the threshold, the allowable
    stress range stops falling. Returns ``(FSR, governing)`` so the caller can
    see which regime applies.

    The metric form, Eq. A-3-1M, uses 6900 rather than 1000 with the same
    ``Cf`` -- so the constants in :data:`TABLE_A3_1` are the US customary ones
    and are not unit-agnostic.

    Raises
    ------
    AISC360Error
        For categories F, C' or C'', which have their own equations.
    """
    if category is StressCategory.F:
        raise AISC360Error(
            "category F uses Eq. A-3-2, not Eq. A-3-1 -- call "
            "category_F_stress_range() instead"
        )
    if category in (StressCategory.C_PRIME, StressCategory.C_DOUBLE_PRIME):
        raise AISC360Error(
            f"category {category.value} has no tabulated Cf: Sects. 3.3(c) and 3.3(d) "
            "compute it from Eqs. A-3-3 or A-3-5 with a reduction factor"
        )
    if nSR <= 0.0:
        raise GeometryError(f"the cycle count must be positive, got {nSR}")

    Cf, FTH = TABLE_A3_1[category]
    computed = 1000.0 * math.pow(Cf / nSR, 0.333)
    if computed >= FTH:
        return computed, f"Eq. A-3-1: finite-life curve governs at {nSR:.3g} cycles"
    return FTH, (
        f"Eq. A-3-1 threshold: FTH = {FTH:g} ksi governs -- indefinite design life"
    )


def category_F_stress_range(nSR: float) -> tuple[Ksi, str]:
    """Allowable stress range for category F.

    AISC 360-16, Eq. A-3-2, Sect. 3.3(b), p. 16.1-197::

        FSR = 100*(1.5/nSR)^0.167  >=  8 ksi

    Category F -- shear on the throat of a fillet or PJP weld -- has a **much
    flatter slope**: the 0.167 exponent against 0.333 everywhere else. Weld
    shear degrades far more slowly with cycle count than base-metal tension, so
    the curve is shallower and the 8 ksi threshold binds later.
    """
    if nSR <= 0.0:
        raise GeometryError(f"the cycle count must be positive, got {nSR}")
    computed = 100.0 * math.pow(1.5 / nSR, 0.167)
    if computed >= 8.0:
        return computed, "Eq. A-3-2: finite-life curve governs"
    return 8.0, "Eq. A-3-2 threshold: 8 ksi governs -- indefinite design life"


def PJP_reduction_factor(twice_a: Inch, w: Inch, tp: Inch) -> float:
    """Reduction factor for a transverse PJP groove weld, crack from the root.

    AISC 360-16, Eq. A-3-4, Sect. 3.3(c), p. 16.1-198::

        RPJP = [0.65 - 0.59*(2a/tp) + 0.72*(w/tp)]/tp^0.1667  <=  1.0

    ``2a`` is the total unwelded root depth and ``w`` the reinforcing or
    contouring fillet leg. The expression is **dimensional** in inches -- the
    metric form, Eq. A-3-4M, uses 1.12, 1.01 and 1.24 with millimetres, which
    are not conversions of these three.

    ``RPJP = 1.0`` means the root is no worse than the toe, and Sect. 3.3(c)
    then says the stress range is limited by the toe and category C.
    """
    for name, value in (("w", w), ("tp", tp)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    if twice_a < 0.0:
        raise GeometryError(f"2a must be non-negative, got {twice_a}")
    numerator = 0.65 - 0.59 * (twice_a / tp) + 0.72 * (w / tp)
    return min(numerator / math.pow(tp, 0.1667), 1.0)


def fillet_reduction_factor(w: Inch, tp: Inch) -> float:
    """Reduction factor for a pair of transverse fillet welds.

    AISC 360-16, Eq. A-3-6, Sect. 3.3(d), p. 16.1-199::

        RFIL = [0.06 + 0.72*(w/tp)]/tp^0.167  <=  1.0

    Simpler than Eq. A-3-4 because a fillet pair has no unwelded root depth to
    account for -- the whole joint is unwelded between the two fillets, and the
    0.06 intercept carries that.

    Note the exponent is printed as ``0.167`` here and ``0.1667`` in Eq. A-3-4.
    The difference is 0.02% on a 1-inch plate and is preserved as printed
    rather than harmonised.
    """
    if w <= 0.0 or tp <= 0.0:
        raise GeometryError(f"w and tp must be positive, got {w}, {tp}")
    return min((0.06 + 0.72 * (w / tp)) / math.pow(tp, 0.167), 1.0)


def PJP_root_stress_range(RPJP: float, nSR: float) -> tuple[Ksi, str]:
    """Allowable stress range at the root of a transverse PJP groove weld.

    AISC 360-16, Eq. A-3-3, Sect. 3.3(c), p. 16.1-198, stress category C'::

        FSR = 1000*RPJP*(4.4/nSR)^0.333

    Note there is **no threshold**: unlike Eq. A-3-1, this curve keeps falling
    for ever. A root-initiated crack has no endurance limit, which is why
    Sect. 3.3(c) requires the toe to be checked as category C as well and the
    lower of the two taken.
    """
    if nSR <= 0.0:
        raise GeometryError(f"the cycle count must be positive, got {nSR}")
    if not 0.0 < RPJP <= 1.0:
        raise GeometryError(f"RPJP must lie in (0, 1], got {RPJP}")
    return (
        1000.0 * RPJP * math.pow(4.4 / nSR, 0.333),
        "Eq. A-3-3, category C': no threshold -- also check the toe as category C",
    )


def fillet_root_stress_range(RFIL: float, nSR: float) -> tuple[Ksi, str]:
    """Allowable stress range at the root of a transverse fillet weld pair.

    AISC 360-16, Eq. A-3-5, Sect. 3.3(d), p. 16.1-199, stress category C''::

        FSR = 1000*RFIL*(4.4/nSR)^0.333

    Identical in form to Eq. A-3-3 with ``RFIL`` in place of ``RPJP``, and with
    the same absence of a threshold. Sect. 3.3(d) notes that if ``RFIL = 1.0``
    the stress range is limited by the weld toe and category C instead.
    """
    if nSR <= 0.0:
        raise GeometryError(f"the cycle count must be positive, got {nSR}")
    if not 0.0 < RFIL <= 1.0:
        raise GeometryError(f"RFIL must lie in (0, 1], got {RFIL}")
    return (
        1000.0 * RFIL * math.pow(4.4 / nSR, 0.333),
        "Eq. A-3-5, category C'': no threshold -- if RFIL = 1.0 use category C",
    )


def bolt_tensile_area(db: Inch, n: float) -> Inch2:
    """Tensile stress area of a bolt.

    AISC 360-16, Eq. A-3-7, Sect. 3.4, p. 16.1-200::

        Atb = (pi/4)*(db - 0.9743/n)^2

    ``n`` is threads per inch. The ``0.9743/n`` deduction is the standard UN
    thread-form correction, which is why this is the **tensile stress area** and
    not the ``pi*db^2/4`` nominal body area that Eq. J3-1 uses. Chapter J puts
    the thread reduction inside ``Fn``; Appendix 3 puts it in the area, and
    mixing the two conventions double-counts or drops it.

    A 3/4-in. UNC bolt (10 threads per inch) gives 0.334 in.^2 against a
    0.442 in.^2 body area -- a 24% difference.

    The metric form, Eq. A-3-7M, uses ``(db - 0.9382*p)`` with ``p`` the pitch
    in millimetres.
    """
    if db <= 0.0 or n <= 0.0:
        raise GeometryError(f"db and n must be positive, got {db}, {n}")
    root = db - 0.9743 / n
    if root <= 0.0:
        raise GeometryError(
            f"db - 0.9743/n = {root:.4g} is not positive: the thread pitch is too "
            f"coarse for a {db:g} in. bolt"
        )
    return math.pi / 4.0 * root**2


def fatigue_need_not_be_considered(
    category: StressCategory, stress_range: Ksi
) -> tuple[bool, str]:
    """Whether fatigue may be disregarded for a detail.

    AISC 360-16, Sect. 3.1, p. 16.1-196, and the ``FTH`` column of Table A-3.1:
    fatigue need not be considered where the calculated stress range is below
    the threshold, regardless of cycle count.

    Sect. 3.1 also excludes fatigue entirely below 20,000 cycles, and states
    that the provisions apply "to structures with adequate corrosion protection"
    and to service-load stresses only.
    """
    if category in (
        StressCategory.F, StressCategory.C_PRIME, StressCategory.C_DOUBLE_PRIME
    ):
        threshold = 8.0 if category is StressCategory.F else 0.0
        if threshold == 0.0:
            return False, (
                f"category {category.value} has no threshold -- Eqs. A-3-3/A-3-5 keep "
                "falling with cycle count and fatigue must always be checked"
            )
    else:
        threshold = TABLE_A3_1[category][1]

    if stress_range < threshold:
        return True, (
            f"stress range {stress_range:g} ksi is below FTH = {threshold:g} ksi for "
            f"category {category.value}: indefinite design life"
        )
    return False, (
        f"stress range {stress_range:g} ksi reaches FTH = {threshold:g} ksi: "
        "fatigue must be evaluated for the design cycle count"
    )
