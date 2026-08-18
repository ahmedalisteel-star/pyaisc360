"""Chapter C -- Design for Stability.

Covers Sects. C1 through C3, pp. 16.1-21 to 16.1-27.

Chapter C is unlike every other chapter in this library: it does not compute a
strength. It changes what the *analysis* must do, and Sect. C3 then routes the
available strength straight back through Chapters D-K "with no further
consideration of overall structure stability".

The direct analysis method makes three changes to the analysis and one to the
member check:

1. **Second-order analysis** (Sect. C2.1) including P-Delta and P-delta, at
   LRFD load level -- or 1.6 times ASD, with the results divided by 1.6
   afterwards.
2. **Initial imperfections** (Sect. C2.2), by modelling them directly or by
   notional loads ``Ni = 0.002*alpha*Yi``.
3. **Reduced stiffness** (Sect. C2.3): ``0.8`` on everything, and an additional
   ``tau_b`` on flexural stiffness once the member is more than half squashed.
4. **Lc = L** (Sect. C3): the effective length is the unbraced length. All of
   the stability that ``K`` used to carry has moved into the analysis.

The payoff is that the direct analysis method has **no limitations** -- it is
"permitted for all structures" -- where both Appendix 7 alternatives are gated
on drift ratios. The price is that the analysis has to be done properly.

The reductions feed Appendix 8 directly: ``EI*`` in Eq. A-8-5 is
``0.8*tau_b*EI`` under this method, which lowers ``Pe1`` and therefore raises
``B1``. Ignoring the reduction there while using the direct analysis method
understates the amplification.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.config import Basis
from .core.enums import StabilityMethod
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.units import Dimensionless, Inch4, Kip, Ksi, Ratio

__all__ = [
    "STIFFNESS_REDUCTION",
    "NOTIONAL_LOAD_COEFFICIENT",
    "TAU_B_THRESHOLD",
    "P_DELTA_DRIFT_LIMIT",
    "OUT_OF_PLUMBNESS",
    "TAU_B_UNITY_NOTIONAL_COEFFICIENT",
    "ReducedStiffness",
    "alpha_for",
    "notional_load",
    "notional_load_for_tau_b_unity",
    "tau_b",
    "reduced_flexural_stiffness",
    "reduced_axial_stiffness",
    "reduced_stiffness",
    "p_delta_may_be_neglected",
    "notional_loads_gravity_only",
    "effective_length_factor",
    "asd_analysis_scale",
]

#: Sect. C2.3(a), p. 16.1-26: "A factor of 0.80 shall be applied to all
#: stiffnesses that are considered to contribute to the stability of the
#: structure." The User Note recommends applying it to *all* members, because
#: reducing some and not others can distort the structure artificially.
STIFFNESS_REDUCTION: float = 0.80

#: Eq. C2-1, p. 16.1-25. Sect. C2.2b(c) records that 0.002 comes from a nominal
#: out-of-plumbness of 1/500 and "where the use of a different maximum
#: out-of-plumbness is justified, it is permissible to adjust the notional load
#: coefficient proportionally".
NOTIONAL_LOAD_COEFFICIENT: float = 0.002
OUT_OF_PLUMBNESS: Ratio = 1.0 / 500.0

#: Sect. C2.3(c), p. 16.1-27: the alternative notional load permitting
#: ``tau_b = 1.0`` throughout. **Additional** to the Eq. C2-1 loads, and not
#: subject to the Sect. C2.2b(d) gravity-only relief.
TAU_B_UNITY_NOTIONAL_COEFFICIENT: float = 0.001

#: Eq. C2-2a/C2-2b, p. 16.1-27: the axial ratio at which tau_b starts to bite.
TAU_B_THRESHOLD: Ratio = 0.5

#: Sects. C2.1(b)(2) and C2.2b(d), pp. 16.1-23 and 16.1-26. Note this is **1.7**,
#: distinct from Appendix 7's 1.5 (whether the alternative methods may be used
#: at all) and 1.1 (whether K = 1.0 skips a sidesway buckling analysis). Three
#: different drift thresholds across the stability provisions, each answering a
#: different question.
P_DELTA_DRIFT_LIMIT: Ratio = 1.7

#: Sect. C2.1(b)(3): no more than one-third of the total gravity load may be
#: carried by columns that are part of moment-resisting frames.
MOMENT_FRAME_GRAVITY_LIMIT: Ratio = 1.0 / 3.0


def alpha_for(basis: Basis) -> float:
    """``alpha`` = 1.0 (LRFD) or 1.6 (ASD).

    AISC 360-16, Sects. C2.2b(a) and C2.3(b), pp. 16.1-25 and 16.1-27.

    Sect. C1 states the principle behind it: "All load-dependent effects shall be
    calculated at a level of loading corresponding to LRFD load combinations or
    1.6 times ASD load combinations." An ASD demand is a service-level force and
    these provisions are calibrated on ultimate-level ones.
    """
    return 1.0 if basis.is_lrfd else 1.6


def asd_analysis_scale(basis: Basis) -> float:
    """Factor by which an ASD second-order analysis must be run and then divided.

    AISC 360-16, Sect. C2.1(d), p. 16.1-24: "For design by ASD, the second-order
    analysis shall be carried out under 1.6 times the ASD load combinations, and
    the results shall be divided by 1.6 to obtain the required strengths."

    The two operations do **not** cancel, because second-order analysis is
    non-linear: amplifying the loads by 1.6, analysing, then dividing by 1.6
    gives a larger answer than analysing the unscaled loads. Skipping the scale
    -- or applying it and forgetting to divide back -- are both wrong, in
    opposite directions.
    """
    return 1.0 if basis.is_lrfd else 1.6


# ===========================================================================
# Sect. C2.2 -- Initial system imperfections
# ===========================================================================
def notional_load(
    Yi: Kip, basis: Basis = Basis.LRFD, *, out_of_plumbness: Ratio = OUT_OF_PLUMBNESS
) -> Kip:
    """Notional load representing initial system imperfections.

    AISC 360-16, Eq. C2-1, Sect. C2.2b(a), p. 16.1-25::

        Ni = 0.002*alpha*Yi,   alpha = 1.0 (LRFD), 1.6 (ASD)

    The notional loads are **additive to other lateral loads** and applied in
    all load combinations, except under the Sect. C2.2b(d) relief. They are not
    an alternative to the wind or seismic case -- they go on top of it.

    ``Yi`` is the gravity load at level ``i``, and the User Note to Sect. C2.1(c)
    stresses that this includes "loads on leaning columns and other elements
    that are not part of the lateral force-resisting system". Omitting leaning
    columns is the usual way this gets understated.

    Parameters
    ----------
    out_of_plumbness:
        Sect. C2.2b(c) permits the 0.002 to be adjusted proportionally where a
        different maximum out-of-plumbness is justified. The default 1/500 is
        the Code of Standard Practice column plumbness tolerance, and 0.002 is
        exactly its reciprocal ratio.
    """
    if Yi < 0.0:
        raise GeometryError(f"gravity load must be non-negative, got {Yi}")
    if out_of_plumbness <= 0.0:
        raise GeometryError(f"out-of-plumbness must be positive, got {out_of_plumbness}")

    coefficient = NOTIONAL_LOAD_COEFFICIENT * (out_of_plumbness / OUT_OF_PLUMBNESS)
    return coefficient * alpha_for(basis) * Yi


def notional_load_for_tau_b_unity(Yi: Kip, basis: Basis = Basis.LRFD) -> Kip:
    """Additional notional load permitting ``tau_b = 1.0`` throughout.

    AISC 360-16, Sect. C2.3(c), p. 16.1-27::

        Ni,additional = 0.001*alpha*Yi

    "In lieu of using tau_b < 1.0 where alpha*Pr/Pns > 0.5, it is permissible to
    use tau_b = 1.0 for all noncomposite members if a notional load of
    0.001*alpha*Yi ... is applied at all levels ... in all load combinations."

    Two conditions travel with it, and both are easy to lose:

    * the load is **added to** the Eq. C2-1 notional loads, not substituted for
      them -- a frame using this option carries ``0.003*alpha*Yi`` in total; and
    * it is **not subject to Sect. C2.2b(d)**, so it applies in every load
      combination including those with lateral load, even where the Eq. C2-1
      loads could be restricted to gravity-only cases.

    The trade is real: one extra notional load in exchange for never having to
    iterate on ``tau_b``, which otherwise depends on ``Pr``, which depends on the
    analysis, which depends on ``tau_b``.
    """
    if Yi < 0.0:
        raise GeometryError(f"gravity load must be non-negative, got {Yi}")
    return TAU_B_UNITY_NOTIONAL_COEFFICIENT * alpha_for(basis) * Yi


def notional_loads_gravity_only(max_drift_ratio: Ratio) -> tuple[bool, str]:
    """Whether notional loads may be restricted to gravity-only combinations.

    AISC 360-16, Sect. C2.2b(d), p. 16.1-26: permitted where the ratio of
    maximum second-order to first-order drift, "with stiffnesses adjusted as
    specified in Section C2.3", is at or below **1.7** in all stories.

    Note the stiffness caveat is the opposite of Appendix 7's: here the ratio is
    judged **with** the reductions applied, where Sects. 7.2.1(b) and 7.3.1(b)
    judge theirs **without** them. Same-looking test, different basis.
    """
    if max_drift_ratio <= P_DELTA_DRIFT_LIMIT:
        return True, (
            f"drift ratio {max_drift_ratio:.3f} <= {P_DELTA_DRIFT_LIMIT}: Sect. C2.2b(d) "
            "permits notional loads in gravity-only combinations"
        )
    return False, (
        f"drift ratio {max_drift_ratio:.3f} > {P_DELTA_DRIFT_LIMIT}: notional loads must be "
        "applied in ALL load combinations"
    )


# ===========================================================================
# Sect. C2.1(b) -- when P-delta may be neglected in the analysis
# ===========================================================================
def p_delta_may_be_neglected(
    max_drift_ratio: Ratio,
    moment_frame_gravity_fraction: Ratio,
    *,
    gravity_through_vertical_elements: bool = True,
) -> tuple[bool, str]:
    """Whether P-delta may be neglected in the *structural* analysis.

    AISC 360-16, Sect. C2.1(b), p. 16.1-23. All three conditions are required:

    1. gravity carried primarily through nominally vertical columns, walls or
       frames;
    2. maximum second-order to first-order drift ratio at or below **1.7** in
       all stories, with stiffnesses adjusted per Sect. C2.3;
    3. no more than **one-third** of the total gravity load carried by columns
       that are part of moment-resisting frames in the direction considered.

    Even when all three hold, Sect. C2.1(b) closes with a sentence that is not
    optional: "It is necessary in all cases to consider P-delta effects in the
    evaluation of individual members subject to compression and flexure." The
    relief is only for the *structure's response*; the member still needs
    ``B1``. The User Note says exactly that -- applying the Appendix 8 ``B1``
    multiplier satisfies the member requirement.
    """
    failures: list[str] = []
    if not gravity_through_vertical_elements:
        failures.append(
            "Sect. C2.1(b)(1): gravity must be carried primarily through nominally "
            "vertical columns, walls or frames"
        )
    if max_drift_ratio > P_DELTA_DRIFT_LIMIT:
        failures.append(
            f"Sect. C2.1(b)(2): drift ratio {max_drift_ratio:.3f} exceeds "
            f"{P_DELTA_DRIFT_LIMIT}"
        )
    if moment_frame_gravity_fraction > MOMENT_FRAME_GRAVITY_LIMIT:
        failures.append(
            f"Sect. C2.1(b)(3): {moment_frame_gravity_fraction:.3f} of the gravity load "
            f"is on moment-frame columns, above the one-third limit"
        )

    member_caveat = (
        "P-delta must STILL be considered for individual members in compression "
        "and flexure -- apply the Appendix 8 B1 multiplier"
    )
    if failures:
        return False, "; ".join(failures)
    return True, (
        "Sect. C2.1(b): P-delta may be neglected in the structure's response. " + member_caveat
    )


# ===========================================================================
# Sect. C2.3 -- Adjustments to stiffness
# ===========================================================================
def tau_b(Pr: Kip, Pns: Kip, basis: Basis = Basis.LRFD) -> Dimensionless:
    """Additional flexural stiffness reduction factor ``tau_b``.

    AISC 360-16, Eqs. C2-2a and C2-2b, Sect. C2.3(b), p. 16.1-27::

        alpha*Pr/Pns <= 0.5:  tau_b = 1.0                          (C2-2a)
        alpha*Pr/Pns >  0.5:  tau_b = 4*(alpha*Pr/Pns)
                                      *[1 - (alpha*Pr/Pns)]        (C2-2b)

    ``Pns`` is the **cross-section** compressive strength -- ``Fy*Ag`` for a
    nonslender-element section, ``Fy*Ae`` for a slender one with ``Ae`` from
    Sect. E7. It is not ``Py``, and it is not the column strength ``Pn``: no
    buckling reduction enters, because ``tau_b`` models *partial yielding of the
    cross section*, accentuated by residual stresses, not member buckling. The
    same quantity appears in Eq. A-7-1.

    Notes
    -----
    Eq. C2-2b is a downward-opening parabola whose **peak sits exactly at the
    transition**: ``4*(0.5)*(0.5) = 1.0`` and its derivative ``4 - 8x`` is zero
    at ``x = 0.5``. So the two branches are continuous in both value and slope
    -- there is no kink at the threshold, which is why no iteration damping is
    needed around it.

    The parabola reaches zero at ``alpha*Pr/Pns = 1.0`` and goes negative
    beyond. A negative flexural stiffness multiplier is meaningless, so the
    result is floored at zero and a ratio above 1.0 raises: the member has
    exceeded its cross-section compressive strength and the design is already
    invalid, not merely soft.

    ``tau_b`` depends on ``Pr``, which comes from an analysis that depends on
    ``tau_b``. Sect. C2.3(c) exists to break that loop --
    :func:`notional_load_for_tau_b_unity` buys ``tau_b = 1.0`` for one extra
    notional load. Where iteration is used instead, the parabola is smooth and
    monotonically decreasing above 0.5, so successive substitution converges
    without damping.

    Raises
    ------
    OutOfScopeError
        If ``alpha*Pr > Pns``. Beyond the cross-section compressive strength
        Eq. C2-2b returns a negative number and the member has no capacity left
        to analyse.
    """
    if Pns <= 0.0:
        raise GeometryError(f"Pns must be positive, got {Pns}")
    if Pr < 0.0:
        # Sect. C2.3(b) applies to compression; a member in tension is not
        # partially yielded in the way tau_b models.
        return 1.0

    ratio = alpha_for(basis) * Pr / Pns
    if ratio <= TAU_B_THRESHOLD:
        return 1.0  # Eq. C2-2a
    if ratio > 1.0:
        raise OutOfScopeError(
            f"alpha*Pr/Pns = {ratio:.4f} exceeds 1.0: the required axial strength is "
            f"above the cross-section compressive strength Pns = {Pns:g} kips. "
            "Eq. C2-2b would return a negative stiffness multiplier -- the member "
            "is inadequate by inspection, not merely soft."
        )
    return max(4.0 * ratio * (1.0 - ratio), 0.0)  # Eq. C2-2b


def reduced_flexural_stiffness(
    E: Ksi, I: Inch4, Pr: Kip, Pns: Kip, basis: Basis = Basis.LRFD, *, tau_b_unity: bool = False
) -> float:
    """Reduced flexural stiffness ``EI* = 0.8*tau_b*EI``.

    AISC 360-16, Sect. C2.3(a) and C2.3(b), pp. 16.1-26 to 16.1-27. The User
    Note there states it directly: "Sections (a) and (b) require the use of
    ``0.8*tau_b`` times the nominal elastic flexural stiffness and 0.8 times
    other nominal elastic stiffnesses".

    This is the ``EI*`` that Eq. A-8-5 asks for. Passing the unreduced ``EI``
    into ``Pe1`` while designing by the direct analysis method overstates
    ``Pe1`` by up to 25% -- and by up to 100% once ``tau_b`` is biting -- which
    understates ``B1``.

    Parameters
    ----------
    tau_b_unity:
        Set when the Sect. C2.3(c) option is being used, where the extra
        ``0.001*alpha*Yi`` notional load buys ``tau_b = 1.0``.
    """
    if E <= 0.0 or I <= 0.0:
        raise GeometryError(f"E and I must be positive, got {E}, {I}")
    factor = 1.0 if tau_b_unity else tau_b(Pr, Pns, basis)
    return STIFFNESS_REDUCTION * factor * E * I


def reduced_axial_stiffness(E: Ksi, A: float) -> float:
    """Reduced axial stiffness ``EA* = 0.8*EA``.

    AISC 360-16, Sect. C2.3(a), p. 16.1-26.

    Note ``tau_b`` does **not** appear: Sect. C2.3(b) applies the additional
    factor to "the flexural stiffnesses of all members", not to axial stiffness.
    Applying ``tau_b`` here as well over-softens the frame.
    """
    if E <= 0.0 or A <= 0.0:
        raise GeometryError(f"E and A must be positive, got {E}, {A}")
    return STIFFNESS_REDUCTION * E * A


@dataclass(frozen=True, slots=True)
class ReducedStiffness:
    """The Sect. C2.3 stiffness adjustments for one member."""

    EI_star: float
    EA_star: float
    tau_b: Dimensionless
    axial_ratio: Ratio
    note: str

    @property
    def flexural_reduction(self) -> float:
        """Combined flexural factor ``0.8*tau_b`` -- 0.8 at most, 0.0 at worst."""
        return STIFFNESS_REDUCTION * self.tau_b

    def report(self, width: int = 78) -> str:
        lines = ["=" * width, "Sect. C2.3 -- adjustments to stiffness", "=" * width]
        lines.append(f"{'alpha*Pr/Pns':<52}{self.axial_ratio:>24,.4f}")
        lines.append(f"{'tau_b (Eq. C2-2a / C2-2b)':<52}{self.tau_b:>24,.4f}")
        lines.append(f"{'flexural factor 0.8*tau_b':<52}{self.flexural_reduction:>24,.4f}")
        lines.append("-" * width)
        lines.append(f"{'EI* = 0.8*tau_b*EI':<52}{self.EI_star:>24,.4g}")
        lines.append(f"{'EA* = 0.8*EA':<52}{self.EA_star:>24,.4g}")
        lines.append(f"    {self.note}")
        lines.append("=" * width)
        return "\n".join(lines)


def reduced_stiffness(
    E: Ksi, I: Inch4, A: float, Pr: Kip, Pns: Kip,
    basis: Basis = Basis.LRFD, *, tau_b_unity: bool = False,
) -> ReducedStiffness:
    """Both Sect. C2.3 stiffness adjustments for one member.

    AISC 360-16, Sect. C2.3, pp. 16.1-26 to 16.1-27.

    ``EI*`` goes into Eq. A-8-5 for ``Pe1``; ``EA*`` goes into the frame model.
    Sect. C2.3(a)'s User Note recommends reducing **all** members, not only
    those contributing to stability, because a partial reduction "can result in
    artificial distortion of the structure under load and possible unintended
    redistribution of forces".
    """
    factor = 1.0 if tau_b_unity else tau_b(Pr, Pns, basis)
    ratio = alpha_for(basis) * max(Pr, 0.0) / Pns if Pns > 0.0 else 0.0

    if tau_b_unity:
        note = (
            "Sect. C2.3(c): tau_b = 1.0 with an additional 0.001*alpha*Yi notional "
            "load at all levels, in ALL load combinations"
        )
    elif factor >= 1.0:
        note = "Eq. C2-2a: alpha*Pr/Pns <= 0.5, no additional flexural reduction"
    else:
        note = f"Eq. C2-2b: partial yielding reduces flexural stiffness to {factor:.3f}"

    return ReducedStiffness(
        EI_star=reduced_flexural_stiffness(E, I, Pr, Pns, basis, tau_b_unity=tau_b_unity),
        EA_star=reduced_axial_stiffness(E, A),
        tau_b=factor,
        axial_ratio=ratio,
        note=note,
    )


# ===========================================================================
# Sect. C3 -- Calculation of available strengths
# ===========================================================================
def effective_length_factor(
    method: StabilityMethod = StabilityMethod.DIRECT_ANALYSIS,
    *,
    K: float | None = None,
) -> float:
    """Effective length factor ``K`` permitted by the active stability method.

    AISC 360-16, Sect. C3, p. 16.1-27: under the direct analysis method "the
    effective length for flexural buckling of all members shall be taken as the
    unbraced length unless a **smaller** value is justified by rational
    analysis".

    So ``K = 1.0`` is not a default to be overridden freely -- it is a ceiling.
    All of the stability that ``K > 1`` used to represent has moved into the
    analysis through the reduced stiffness and notional loads, and taking it
    again in the member check would double-count. Appendix 7.3.3 says the same
    for the first-order analysis method.

    Under the **effective length method** the opposite holds: ``K`` comes from a
    sidesway buckling analysis and may well exceed 1.0, which is the whole point
    of that method. Sect. 7.2.3 governs there, and this function defers to the
    supplied value.

    Raises
    ------
    OutOfScopeError
        If ``K > 1.0`` is supplied under the direct analysis or first-order
        method -- both fix the effective length at the unbraced length.
    """
    if method is StabilityMethod.EFFECTIVE_LENGTH:
        if K is None:
            raise AISC360Error(
                "the effective length method needs K from a sidesway buckling "
                "analysis (Sect. 7.2.3(b)), or K = 1.0 for a braced frame"
            )
        if K <= 0.0:
            raise GeometryError(f"K must be positive, got {K}")
        return K

    if K is None:
        return 1.0
    if K <= 0.0:
        raise GeometryError(f"K must be positive, got {K}")
    if K > 1.0:
        raise OutOfScopeError(
            f"K = {K} exceeds 1.0 under the {method.value}. Sect. C3 (and Sect. "
            "7.3.3) take Lc as the unbraced length unless a SMALLER value is "
            "justified -- the stability K used to carry is already in the analysis, "
            "and taking it again double-counts."
        )
    return K
