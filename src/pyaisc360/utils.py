"""Shared mechanics used by more than one chapter.

Anything here is called from at least two chapters.  Provisions specific to a
single chapter live in that chapter's module, even when they look generic --
keeping Chapter F's business out of ``utils`` is what stops this file becoming
the place where the Specification's structure goes to die.
"""

from __future__ import annotations

import math

from .core.config import Basis
from .core.enums import AxialSlenderness, FlexuralSlenderness
from .core.exceptions import AISC360Error, GeometryError
from .core.units import Dimensionless, KipIn, Ksi, Ratio

__all__ = [
    "available_strength",
    "elastic_buckling_stress",
    "flexural_buckling_stress",
    "classify_axial_element",
    "classify_flexural_element",
    "lateral_torsional_modification_factor",
    "limiting_ratio",
    "linear_transition",
    "governing_slenderness",
]

# The Specification's factors are calibrated so that Omega = 1.5/phi throughout
# (phi 0.90 -> Omega 1.67, phi 0.75 -> Omega 2.00).  Used to catch a transposed
# factor pair, which is otherwise invisible: both numbers look reasonable alone.
_CALIBRATION = 1.5
_CALIBRATION_TOL = 0.02


# ---------------------------------------------------------------------------
# 1. The LRFD / ASD gate -- every chapter returns through this
# ---------------------------------------------------------------------------
def available_strength(
    Rn: float,
    *,
    phi: float,
    omega: float,
    basis: Basis,
    check_calibration: bool = True,
) -> float:
    """Available strength: ``phi*Rn`` (LRFD) or ``Rn/Omega`` (ASD).

    AISC 360-16, Eq. B3-1 (LRFD) and Eq. B3-2 (ASD), Sect. B3.1-B3.2,
    p. 16.1-12.

    Both factors are always supplied so the same nominal strength can be read
    under either philosophy; only ``basis`` decides which is applied.

    Parameters
    ----------
    Rn:
        Nominal strength, in any consistent unit.
    phi:
        Resistance factor for the governing limit state (LRFD).
    omega:
        Safety factor for the governing limit state (ASD).
    basis:
        Which of Eq. B3-1 / B3-2 to apply.
    check_calibration:
        Verify the factor pair is a real Specification pair: ``phi <= 1``,
        ``omega >= 1``, and ``phi * omega ~ 1.5``.

        The product alone cannot catch a swapped pair -- multiplication is
        commutative, so ``phi=1.67, omega=0.90`` passes it. The bounds are what
        actually catch the transposition, and they hold for every factor pair in
        the Specification (phi never exceeds 1.00, at Sect. G2.1(a); omega never
        falls below 1.50).

    Raises
    ------
    AISC360Error
        On a negative nominal strength, a non-positive factor, or -- when
        ``check_calibration`` is set -- a factor pair that is not a valid
        Specification pair.
    """
    if Rn < 0.0:
        raise AISC360Error(f"nominal strength must be non-negative, got {Rn}")
    if phi <= 0.0 or omega <= 0.0:
        raise AISC360Error(f"phi and omega must be positive, got phi={phi}, omega={omega}")
    if check_calibration:
        if phi > 1.0 or omega < 1.0:
            raise AISC360Error(
                f"phi={phi} and omega={omega} are not a valid factor pair "
                "(the Specification uses phi <= 1.00 and omega >= 1.50). "
                "The arguments look transposed."
            )
        if abs(phi * omega - _CALIBRATION) > _CALIBRATION_TOL:
            raise AISC360Error(
                f"phi={phi} and omega={omega} are not mutually calibrated "
                f"(phi*omega = {phi * omega:.3f}, expected ~{_CALIBRATION})."
            )
    return phi * Rn if basis.is_lrfd else Rn / omega


# ---------------------------------------------------------------------------
# 2. Elastic (Euler) buckling stress
# ---------------------------------------------------------------------------
def elastic_buckling_stress(slenderness: Ratio, E: Ksi) -> Ksi:
    """Elastic buckling stress ``Fe = pi^2*E / (Lc/r)^2``.

    AISC 360-16, Eq. E3-4, Sect. E3, p. 16.1-36.

    Parameters
    ----------
    slenderness:
        Effective slenderness ratio ``Lc/r``, where ``Lc = KL`` is the effective
        length (Sect. E2, p. 16.1-35). Dimensionless.
    E:
        Modulus of elasticity, ksi. 29,000 ksi for structural steel.

    Notes
    -----
    Sect. E2 carries a User Note that ``Lc/r`` preferably should not exceed 200.
    That is advisory, not a limit, so it is not enforced here -- Chapter E
    surfaces it as a note on the result instead.

    Raises
    ------
    GeometryError
        If the slenderness ratio is not positive. A zero ratio means a zero
        effective length, for which Fe is unbounded and Chapter E does not apply.
    """
    if slenderness <= 0.0:
        raise GeometryError(
            f"slenderness ratio Lc/r must be positive, got {slenderness}; "
            "a fully braced member has no flexural buckling limit state"
        )
    return math.pi**2 * E / slenderness**2


# ---------------------------------------------------------------------------
# 3. Critical stress -- the inelastic/elastic transition
# ---------------------------------------------------------------------------
def flexural_buckling_stress(Fy: Ksi, Fe: Ksi) -> Ksi:
    """Critical stress ``Fcr`` from the yield stress and elastic buckling stress.

    AISC 360-16, Eq. E3-2 and Eq. E3-3, Sect. E3, p. 16.1-35.

    (a) When ``Fy/Fe <= 2.25``  ->  ``Fcr = 0.658**(Fy/Fe) * Fy``   (Eq. E3-2)
    (b) When ``Fy/Fe >  2.25``  ->  ``Fcr = 0.877 * Fe``            (Eq. E3-3)

    The ``Fy/Fe <= 2.25`` test is the algebraic twin of ``Lc/r <= 4.71*sqrt(E/Fy)``;
    the User Note on p. 16.1-36 confirms the two give the same result for
    flexural buckling. The stress form is used here because it also serves the
    torsional and flexural-torsional cases of Sect. E4 and the slender-element
    provisions of Sect. E7, which have no single ``Lc/r`` to test.

    The two are equivalent because ``Lc/r = pi*sqrt(E/Fe)``, so ``Fy/Fe = 2.25``
    at ``Lc/r = pi*sqrt(2.25*E/Fy) = 4.7124*sqrt(E/Fy)``. The printed 4.71 is
    that value rounded, which puts the two criteria about 0.05% apart -- close
    enough that no real section lands between them, but not identical. Using the
    stress form throughout means the library picks one side consistently.

    Parameters
    ----------
    Fy:
        Specified minimum yield stress, ksi. For Sect. E7 this is replaced by
        the local-buckling stress in the caller, not here.
    Fe:
        Elastic buckling stress, ksi -- from Eq. E3-4 for flexural buckling, or
        Eqs. E4-2 through E4-7 for torsional and flexural-torsional buckling.

    Raises
    ------
    AISC360Error
        If either stress is non-positive.
    """
    if Fy <= 0.0:
        raise AISC360Error(f"Fy must be positive, got {Fy}")
    if Fe <= 0.0:
        raise AISC360Error(f"Fe must be positive, got {Fe}")

    if Fy / Fe <= 2.25:
        # math.pow rather than ** so the result is a float, not Any: mypy widens
        # ** to Any because a negative base with a fractional exponent is complex.
        return math.pow(0.658, Fy / Fe) * Fy  # Eq. E3-2
    return 0.877 * Fe  # Eq. E3-3


# ---------------------------------------------------------------------------
# 4. Width-to-thickness classification
# ---------------------------------------------------------------------------
def classify_axial_element(lam: Ratio, lam_r: Ratio) -> AxialSlenderness:
    """Classify a compression element under **axial** load, per Table B4.1a.

    AISC 360-16, Sect. B4.1, p. 16.1-16; limits from Table B4.1a, p. 16.1-16.

    Axial compression has only two classes: nonslender (``lambda <= lambda_r``)
    or slender. There is no "compact" here -- that concept belongs to flexure
    only, which is why this returns a different enum from
    :func:`classify_flexural_element`.

    Parameters
    ----------
    lam:
        The element's actual width-to-thickness ratio, ``b/t``, ``h/tw``, ``D/t``.
    lam_r:
        Limiting ratio separating nonslender from slender, from Table B4.1a.
    """
    _validate_limits(lam, lam_r=lam_r)
    return AxialSlenderness.NONSLENDER if lam <= lam_r else AxialSlenderness.SLENDER


def classify_flexural_element(lam: Ratio, lam_p: Ratio, lam_r: Ratio) -> FlexuralSlenderness:
    """Classify a compression element under **flexure**, per Table B4.1b.

    AISC 360-16, Sect. B4.1, p. 16.1-16; limits from Table B4.1b, p. 16.1-17.

    compact     ``lambda <= lambda_p``
    noncompact  ``lambda_p < lambda <= lambda_r``
    slender     ``lambda > lambda_r``

    Sect. B4.1 additionally requires that for a section to qualify as compact
    its flanges be *continuously connected* to the web. That is a fabrication
    condition, not a ratio, so it is checked by the section adapter rather than
    here.

    Parameters
    ----------
    lam:
        The element's actual width-to-thickness ratio.
    lam_p:
        Limiting ratio for a compact element.
    lam_r:
        Limiting ratio for a noncompact element.
    """
    _validate_limits(lam, lam_p=lam_p, lam_r=lam_r)
    if lam <= lam_p:
        return FlexuralSlenderness.COMPACT
    if lam <= lam_r:
        return FlexuralSlenderness.NONCOMPACT
    return FlexuralSlenderness.SLENDER


def governing_slenderness(*classes: FlexuralSlenderness) -> FlexuralSlenderness:
    """Worst classification among a section's elements.

    Sect. B4.1: a section is compact only if *every* compression element is
    compact, and is slender-element if *any* element is slender.
    """
    if not classes:
        raise AISC360Error("no element classifications supplied")
    order = {
        FlexuralSlenderness.COMPACT: 0,
        FlexuralSlenderness.NONCOMPACT: 1,
        FlexuralSlenderness.SLENDER: 2,
    }
    return max(classes, key=lambda c: order[c])


def _validate_limits(lam: Ratio, *, lam_r: Ratio, lam_p: Ratio | None = None) -> None:
    if lam < 0.0:
        raise GeometryError(f"width-to-thickness ratio must be non-negative, got {lam}")
    if lam_r <= 0.0:
        raise AISC360Error(f"lambda_r must be positive, got {lam_r}")
    if lam_p is not None:
        if lam_p <= 0.0:
            raise AISC360Error(f"lambda_p must be positive, got {lam_p}")
        if lam_p >= lam_r:
            raise AISC360Error(
                f"lambda_p ({lam_p}) must be less than lambda_r ({lam_r}); "
                "the arguments look transposed"
            )


# ---------------------------------------------------------------------------
# 5. Lateral-torsional buckling modification factor
# ---------------------------------------------------------------------------
def lateral_torsional_modification_factor(
    Mmax: KipIn,
    MA: KipIn,
    MB: KipIn,
    MC: KipIn,
    *,
    cantilever: bool = False,
) -> Dimensionless:
    """Lateral-torsional buckling modification factor ``Cb``.

    AISC 360-16, Eq. F1-1, Sect. F1(c), p. 16.1-46::

        Cb = 12.5*Mmax / (2.5*Mmax + 3*MA + 4*MB + 3*MC)

    All four moments are **absolute values** taken over the unbraced segment,
    with both ends of the segment braced.

    Parameters
    ----------
    Mmax:
        Maximum moment in the unbraced segment.
    MA, MB, MC:
        Moments at the quarter point, centreline and three-quarter point of the
        unbraced segment.
    cantilever:
        Set for a cantilever where warping is prevented at the support and the
        free end is unbraced; Sect. F1(c) then requires ``Cb = 1.0``.

    Notes
    -----
    360-16 places no upper cap on ``Cb`` in Eq. F1-1, and none is imposed here.
    Taking ``Cb = 1.0`` is always permitted and always conservative.

    The User Note on p. 16.1-46 gives the checks this reproduces for doubly
    symmetric members with no transverse load between brace points: 1.0 for
    uniform moment, 2.27 for equal end moments in reverse curvature, and 1.67
    when one end moment is zero.

    Raises
    ------
    GeometryError
        If ``Mmax`` is not the largest of the four moments -- which means the
        quarter-point sampling missed the peak, and ``Cb`` would be overstated.
    """
    if cantilever:
        return 1.0

    Mmax, MA, MB, MC = abs(Mmax), abs(MA), abs(MB), abs(MC)

    if Mmax == 0.0:
        raise GeometryError("Mmax is zero; Cb is undefined for an unloaded segment")
    if Mmax < max(MA, MB, MC):
        raise GeometryError(
            f"Mmax ({Mmax:g}) is smaller than a quarter-point moment "
            f"(max {max(MA, MB, MC):g}); Mmax must be the maximum in the segment"
        )

    return 12.5 * Mmax / (2.5 * Mmax + 3.0 * MA + 4.0 * MB + 3.0 * MC)


# ---------------------------------------------------------------------------
# Runners-up: called ~20 times each across Chapters B, E, F and G
# ---------------------------------------------------------------------------
def limiting_ratio(coefficient: float, E: Ksi, Fy: Ksi) -> Ratio:
    """``coefficient * sqrt(E/Fy)`` -- the form nearly every limit takes.

    Covers the 4.71 of Sect. E3, the 0.56/1.0/3.76/5.70 of Tables B4.1a/b, the
    1.76 of Eq. F2-5, and the 0.71 of Sect. E5. Keeping one implementation means
    the square root is evaluated identically everywhere, so the compact/slender
    boundary never lands on different sides of a limit in two chapters.

    Parameters
    ----------
    coefficient:
        The dimensionless multiplier printed in the governing table or equation.
    E, Fy:
        Modulus of elasticity and yield stress, in the same stress unit.
    """
    if Fy <= 0.0:
        raise AISC360Error(f"Fy must be positive, got {Fy}")
    if E <= 0.0:
        raise AISC360Error(f"E must be positive, got {E}")
    return coefficient * math.sqrt(E / Fy)


def linear_transition(
    upper: float,
    lower: float,
    lam: Ratio,
    lam_p: Ratio,
    lam_r: Ratio,
) -> float:
    """Straight-line interpolation across the noncompact range.

    This is the shape shared by Eqs. F3-1, F4-13, F7-2, F7-6, F9-6 and others::

        value = upper - (upper - lower) * (lam - lam_p) / (lam_r - lam_p)

    Each of those equations is cited at its own call site; this helper only
    performs the arithmetic, and deliberately carries no citation of its own
    because no single equation number owns the pattern.

    The result is clamped to ``[lower, upper]`` so a ``lam`` marginally outside
    the noncompact band -- which happens with rounded catalog ratios -- returns
    the endpoint rather than extrapolating past it.
    """
    if lam_r <= lam_p:
        raise AISC360Error(f"lambda_r ({lam_r}) must exceed lambda_p ({lam_p})")
    fraction = (lam - lam_p) / (lam_r - lam_p)
    fraction = min(max(fraction, 0.0), 1.0)
    return upper - (upper - lower) * fraction
