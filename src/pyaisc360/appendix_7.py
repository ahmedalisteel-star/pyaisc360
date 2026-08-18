"""Appendix 7 -- Alternative Methods of Design for Stability.

Covers Sects. 7.1 through 7.3, pp. 16.1-245 to 16.1-248.

Two alternatives to the direct analysis method of Chapter C:

**Effective length method** (Sect. 7.2) -- analyse with *nominal* stiffness (no
    ``0.8*tau_b`` reduction), apply notional loads in gravity-only cases, and
    carry the stability into the member check through ``K``.

**First-order analysis method** (Sect. 7.3) -- analyse first-order only, absorb
    stability into an additional lateral load ``Ni``, take ``Lc = L``, and
    amplify beam-column moments with ``B1``.

Both are gated on the same drift limit: the second-order to first-order drift
ratio must not exceed **1.5** in any story. Appendix 7's own User Note says that
ratio "may be taken as the B2 multiplier, calculated as specified in Appendix
8" -- so :func:`pyaisc360.appendix_8.B2_multiplier` is the practical test, and
the two appendices are used together.

Only two numbered equations live here, both in Sect. 7.3; the rest of the
appendix is procedural requirements, which are returned as explicit check
results rather than silently assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.config import Basis
from .core.enums import StabilityMethod
from .core.exceptions import GeometryError, OutOfScopeError
from .core.units import Kip, Ratio

__all__ = [
    "DRIFT_RATIO_LIMIT",
    "K_EQUAL_ONE_DRIFT_LIMIT",
    "AXIAL_LIMIT_FRACTION",
    "MINIMUM_NOTIONAL_FRACTION",
    "StabilityMethod",
    "MethodCheck",
    "alpha_for",
    "effective_length_method_permitted",
    "K_may_be_taken_as_unity",
    "first_order_method_permitted",
    "axial_force_limit",
    "additional_lateral_load",
]

#: Sects. 7.2.1(b) and 7.3.1(b): max second-order / first-order drift ratio in
#: any story, for either alternative method to be used at all.
DRIFT_RATIO_LIMIT: Ratio = 1.5

#: Sect. 7.2.3(b) Exception, p. 16.1-246: below this drift ratio, K = 1.0 is
#: permitted for *all* columns even in moment frames -- which removes the need
#: for a sidesway buckling analysis entirely.
K_EQUAL_ONE_DRIFT_LIMIT: Ratio = 1.1

#: Eq. A-7-1: the required axial strength must not exceed half the cross-section
#: compressive strength.
AXIAL_LIMIT_FRACTION: float = 0.5

#: Eq. A-7-2: the notional load floor, as a fraction of the level's gravity load.
MINIMUM_NOTIONAL_FRACTION: float = 0.0042


def alpha_for(basis: Basis) -> float:
    """``alpha`` = 1.0 (LRFD) or 1.6 (ASD). Sects. 7.3.1(c) and 7.3.2(a)."""
    return 1.0 if basis.is_lrfd else 1.6


@dataclass(frozen=True, slots=True)
class MethodCheck:
    """Whether a method's limitations are satisfied, and which failed."""

    permitted: bool
    method: StabilityMethod
    failures: tuple[str, ...] = ()
    note: str = ""

    def require(self) -> None:
        """Raise if the method is not permitted."""
        if not self.permitted:
            raise OutOfScopeError(
                f"{self.method.value} is not permitted: " + "; ".join(self.failures)
            )


# ===========================================================================
# Sect. 7.2 -- Effective Length Method
# ===========================================================================
def effective_length_method_permitted(
    max_drift_ratio: Ratio, *, gravity_through_vertical_elements: bool = True
) -> MethodCheck:
    """Sect. 7.2.1 limitations, p. 16.1-245.

    (a) The structure supports gravity loads primarily through nominally
        vertical columns, walls or frames.
    (b) The maximum second-order to first-order drift ratio in **all** stories
        is at or below 1.5, with stiffness **not** adjusted per Sect. C2.3.

    The stiffness caveat in (b) is easy to miss: the ratio is judged on nominal
    stiffness, not the reduced stiffness the direct analysis method would use.

    Sect. 7.2.2 then requires the analysis to conform to Sect. C2.1 *except*
    that the Sect. C2.1(a) stiffness reduction is not applied, with notional
    loads per Sect. C2.2b -- and the User Note records that those need only be
    applied in gravity-only load cases.
    """
    failures: list[str] = []
    if not gravity_through_vertical_elements:
        failures.append(
            "Sect. 7.2.1(a): gravity must be carried primarily through nominally "
            "vertical columns, walls or frames"
        )
    if max_drift_ratio > DRIFT_RATIO_LIMIT:
        failures.append(
            f"Sect. 7.2.1(b): max second-order/first-order drift ratio "
            f"{max_drift_ratio:.3f} exceeds {DRIFT_RATIO_LIMIT}"
        )
    return MethodCheck(
        permitted=not failures,
        method=StabilityMethod.EFFECTIVE_LENGTH,
        failures=tuple(failures),
        note=(
            "Sect. 7.2.2: analyse with NOMINAL stiffness (no 0.8*tau_b reduction), "
            "notional loads per Sect. C2.2b in gravity-only cases"
        ),
    )


def K_may_be_taken_as_unity(max_drift_ratio: Ratio, *, braced_frame: bool = False) -> bool:
    """Whether ``K = 1.0`` is permitted for all columns.

    AISC 360-16, Sect. 7.2.3, p. 16.1-246.

    (a) In braced-frame and shear-wall systems -- and any system where lateral
        stability does not rely on column flexural stiffness -- ``K = 1.0``
        always, "unless a smaller value is justified by rational analysis".
    (b) In moment-frame systems, ``K`` comes from a sidesway buckling analysis,
        **except** that ``K = 1.0`` is permitted for all columns if the maximum
        second-order to first-order drift ratio in all stories is at or below
        **1.1**.

    Note the two different drift thresholds in this appendix: 1.5 gates whether
    the method may be used at all, 1.1 gates whether the sidesway buckling
    analysis can be skipped.
    """
    if braced_frame:
        return True
    return max_drift_ratio <= K_EQUAL_ONE_DRIFT_LIMIT


# ===========================================================================
# Sect. 7.3 -- First-Order Analysis Method
# ===========================================================================
def axial_force_limit(Pns: Kip) -> Kip:
    """Maximum ``alpha*Pr`` for the first-order analysis method.

    AISC 360-16, Eq. A-7-1, Sect. 7.3.1(c), p. 16.1-247::

        alpha*Pr <= 0.5*Pns

    ``Pns`` is the **cross-section** compressive strength -- ``Fy*Ag`` for a
    nonslender-element section, or ``Fy*Ae`` for a slender-element one with
    ``Ae`` from Sect. E7. It is not the column strength ``Pn``: no buckling
    reduction enters, because this limit is about how far into inelasticity the
    member is, not how it buckles.

    The limit applies to "all members whose flexural stiffnesses are considered
    to contribute to the lateral stability of the structure".
    """
    if Pns <= 0.0:
        raise GeometryError(f"Pns must be positive, got {Pns}")
    return AXIAL_LIMIT_FRACTION * Pns


def first_order_method_permitted(
    max_drift_ratio: Ratio,
    Pr: Kip,
    Pns: Kip,
    basis: Basis = Basis.LRFD,
    *,
    gravity_through_vertical_elements: bool = True,
) -> MethodCheck:
    """Sect. 7.3.1 limitations, p. 16.1-246.

    (a) Gravity carried primarily through nominally vertical elements.
    (b) Maximum second-order to first-order drift ratio at or below 1.5.
    (c) ``alpha*Pr <= 0.5*Pns`` for every member contributing flexural stiffness
        to lateral stability (Eq. A-7-1).

    Condition (c) is the one the effective length method does not have: the
    first-order method absorbs second-order effects into a fixed notional load,
    which is only calibrated while the columns are well below their squash load.
    """
    failures: list[str] = []
    if not gravity_through_vertical_elements:
        failures.append("Sect. 7.3.1(a): gravity must be carried through vertical elements")
    if max_drift_ratio > DRIFT_RATIO_LIMIT:
        failures.append(
            f"Sect. 7.3.1(b): drift ratio {max_drift_ratio:.3f} exceeds {DRIFT_RATIO_LIMIT}"
        )
    demand = alpha_for(basis) * Pr
    limit = axial_force_limit(Pns)
    if demand > limit:
        failures.append(
            f"Eq. A-7-1: alpha*Pr = {demand:.4g} exceeds 0.5*Pns = {limit:.4g}"
        )
    return MethodCheck(
        permitted=not failures,
        method=StabilityMethod.FIRST_ORDER,
        failures=tuple(failures),
        note=(
            "Sect. 7.3.2(b): the B1 amplifier of Appendix 8 must still be applied to "
            "the TOTAL member moments; Sect. 7.3.3: Lc = L unless analysis justifies less"
        ),
    )


def additional_lateral_load(
    Yi: Kip, drift_over_height: Ratio, basis: Basis = Basis.LRFD
) -> tuple[Kip, str]:
    """Additional lateral load ``Ni`` for the first-order analysis method.

    AISC 360-16, Eq. A-7-2, Sect. 7.3.2(a), p. 16.1-247::

        Ni = 2.1*alpha*(Delta/L)*Yi  >= 0.0042*Yi

    ``Delta/L`` is the **maximum** first-order interstory drift ratio over all
    stories in the structure -- not this level's own drift. One flexible story
    therefore raises the notional load everywhere.

    Returns ``(Ni, governing)``. The ``0.0042*Yi`` floor is the same notional
    load Sect. C2.2b applies under the direct analysis method, so a stiff frame
    lands on the familiar 0.2% value.

    Sect. 7.3.2(a) also requires ``Ni`` to be "distributed over that level in
    the same manner as the gravity load" and "applied in the direction that
    provides the greatest destabilizing effect" -- both layout decisions this
    function cannot make.
    """
    if Yi < 0.0:
        raise GeometryError(f"gravity load must be non-negative, got {Yi}")
    if drift_over_height < 0.0:
        raise GeometryError(f"drift ratio must be non-negative, got {drift_over_height}")

    computed = 2.1 * alpha_for(basis) * drift_over_height * Yi
    floor = MINIMUM_NOTIONAL_FRACTION * Yi
    if computed >= floor:
        return computed, "Eq. A-7-2, drift-based term governs"
    return floor, "Eq. A-7-2 floor: 0.0042*Yi governs (same as Sect. C2.2b)"
