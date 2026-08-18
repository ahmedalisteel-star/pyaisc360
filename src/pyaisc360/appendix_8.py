"""Appendix 8 -- Approximate Second-Order Analysis.

Covers Sects. 8.1 and 8.2, pp. 16.1-249 to 16.1-251.

Two amplifiers applied to the results of two first-order analyses:

``B1`` -- **P-delta**, the member effect. Amplifies the moment a compression
    member develops because its own deflection acts at an eccentricity to the
    axial force. Per member, per direction of bending.

``B2`` -- **P-Delta**, the story effect. Amplifies moments *and* axial forces
    arising from lateral translation, because the whole story's gravity load
    acts at the drifted position. Per story, per direction of translation.

The split matters: ``B1`` multiplies only ``Mnt`` (the no-translation moment)
and ``B2`` multiplies only ``Mlt`` (the translation moment) and ``Plt``. A
common error is applying ``B2`` to the whole moment, which over-amplifies the
gravity part, or applying ``B1`` to ``Mlt``, which double-counts.

``alpha`` is 1.0 for LRFD and 1.6 for ASD throughout, because these expressions
are calibrated on ultimate-level forces and an ASD demand is at service level.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .core.config import Basis
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.units import Dimensionless, Inch, Kip, KipIn, Ratio

__all__ = [
    "RM_BRACED",
    "RM_MOMENT_FRAME_LOWER_BOUND",
    "AmplifiedForces",
    "alpha_for",
    "Cm_no_transverse_load",
    "Pe1",
    "B1_multiplier",
    "story_buckling_strength",
    "RM_factor",
    "B2_multiplier",
    "amplified_forces",
]

#: Eq. A-8-8 User Note, p. 16.1-252: RM = 1 for a story with no moment frames.
RM_BRACED: float = 1.0
#: The same User Note: 0.85 is a lower-bound RM for stories including moment frames.
RM_MOMENT_FRAME_LOWER_BOUND: float = 0.85


def alpha_for(basis: Basis) -> float:
    """``alpha`` = 1.0 (LRFD) or 1.6 (ASD). Sects. 8.2.1 and 8.2.2."""
    return 1.0 if basis.is_lrfd else 1.6


# ===========================================================================
# Sect. 8.2.1 -- B1, the P-delta multiplier
# ===========================================================================
def Cm_no_transverse_load(M1_over_M2: Ratio) -> Dimensionless:
    """Equivalent uniform moment factor for a member without transverse load.

    AISC 360-16, Eq. A-8-4, Sect. 8.2.1, p. 16.1-250::

        Cm = 0.6 - 0.4*(M1/M2)

    ``M1`` and ``M2`` are the smaller and larger first-order end moments of the
    portion of the member unbraced in the plane of bending. ``M1/M2`` is
    **positive when the member is bent in reverse curvature** and negative in
    single curvature -- the same convention Eq. F13-8 uses.

    The sign is the whole point: single curvature (negative ratio) puts the peak
    moment near midspan where the P-delta deflection is largest, giving
    ``Cm`` up to 1.0; reverse curvature (positive) keeps the peak at the ends
    where the deflection is zero, dropping ``Cm`` to as little as 0.2.

    For a member **with** transverse loading between supports, Sect. 8.2.1(b)
    requires ``Cm`` from analysis or conservatively 1.0 -- this function does not
    apply, and passing an end-moment ratio for such a member understates ``B1``.
    """
    if not -1.0 <= M1_over_M2 <= 1.0:
        raise GeometryError(
            f"M1/M2 = {M1_over_M2} is outside [-1, 1]; M1 is the SMALLER end moment"
        )
    return 0.6 - 0.4 * M1_over_M2


def Pe1(EI_star: float, Lc1: Inch) -> Kip:
    """Elastic critical buckling strength in the plane of bending, no translation.

    AISC 360-16, Eq. A-8-5, Sect. 8.2.1, p. 16.1-250::

        Pe1 = pi^2*EI* / Lc1^2

    ``EI*`` is "the flexural rigidity required to be used in the analysis":
    ``0.8*tau_b*EI`` under the direct analysis method (Sect. C2.3), or the
    unreduced ``EI`` for the effective length and first-order analysis methods.
    Passing the unreduced ``EI`` while running the direct analysis method
    overstates ``Pe1`` by up to 25% and understates ``B1``.

    ``Lc1`` is the effective length in the plane of bending assuming no lateral
    translation -- set equal to the unbraced length unless analysis justifies
    less. It is *not* the sway effective length.
    """
    if EI_star <= 0.0:
        raise GeometryError(f"EI* must be positive, got {EI_star}")
    if Lc1 <= 0.0:
        raise GeometryError(f"Lc1 must be positive, got {Lc1}")
    return math.pi**2 * EI_star / Lc1**2


def B1_multiplier(Cm: Dimensionless, Pr: Kip, Pe1_value: Kip, basis: Basis) -> Dimensionless:
    """P-delta amplifier for one member and one direction of bending.

    AISC 360-16, Eq. A-8-3, Sect. 8.2.1, p. 16.1-250::

        B1 = Cm/(1 - alpha*Pr/Pe1)  >= 1

    The ``>= 1`` floor matters: a low ``Cm`` from strong reverse curvature would
    otherwise return an amplifier below unity and *reduce* the design moment,
    which the equation never intends.

    Sect. 8.2.1 permits the first-order estimate ``Pr = Pnt + Plt`` here, so no
    iteration is needed.

    Sect. 8.2 also states ``B1`` is taken as 1.0 for members not subject to
    compression -- so a tension member never amplifies.

    Raises
    ------
    OutOfScopeError
        If ``alpha*Pr >= Pe1``. The member has reached its in-plane elastic
        buckling load; the amplifier is singular and the section is inadequate
        by inspection, not by a large number.
    """
    if Cm <= 0.0:
        raise GeometryError(f"Cm must be positive, got {Cm}")
    if Pe1_value <= 0.0:
        raise GeometryError(f"Pe1 must be positive, got {Pe1_value}")
    if Pr <= 0.0:
        return 1.0  # Sect. 8.2: B1 = 1.0 for members not subject to compression

    demand = alpha_for(basis) * Pr / Pe1_value
    if demand >= 1.0:
        raise OutOfScopeError(
            f"alpha*Pr/Pe1 = {demand:.4f} reaches 1.0: the member is at or beyond "
            "its in-plane elastic buckling load, so Eq. A-8-3 is singular. Increase "
            "the section or reduce the effective length."
        )
    return max(Cm / (1.0 - demand), 1.0)


# ===========================================================================
# Sect. 8.2.2 -- B2, the P-Delta multiplier
# ===========================================================================
def RM_factor(Pmf: Kip, Pstory: Kip) -> Dimensionless:
    """Story moment-frame reduction factor.

    AISC 360-16, Eq. A-8-8, Sect. 8.2.2, p. 16.1-252::

        RM = 1 - 0.15*(Pmf/Pstory)

    ``Pmf`` is the vertical load in columns that are part of moment frames --
    **zero for a braced-frame system**, which makes ``RM = 1.0``. The User Note
    gives 0.85 as a lower bound for stories that include moment frames, which is
    the value at ``Pmf = Pstory``.

    The factor accounts for the extra flexibility a moment frame's columns
    contribute to sidesway; a braced frame's columns are not bending, so no
    reduction applies.
    """
    if Pstory <= 0.0:
        raise GeometryError(f"Pstory must be positive, got {Pstory}")
    if Pmf < 0.0 or Pmf > Pstory:
        raise GeometryError(f"need 0 <= Pmf <= Pstory, got Pmf={Pmf}, Pstory={Pstory}")
    return 1.0 - 0.15 * (Pmf / Pstory)


def story_buckling_strength(RM: Dimensionless, H: Kip, L: Inch, drift: Inch) -> Kip:
    """Elastic critical buckling strength for the story.

    AISC 360-16, Eq. A-8-7, Sect. 8.2.2, p. 16.1-252::

        Pe,story = RM*H*L/delta_H

    ``H/delta_H`` is the story's lateral stiffness, so this is a stiffness times
    a height. The User Note records that ``H`` and ``delta_H`` "may be based on
    any lateral loading that provides a representative value of story lateral
    stiffness" -- the ratio is what matters, not the magnitudes.

    ``delta_H`` must be computed "using the stiffness required to be used in the
    analysis", i.e. the reduced stiffness under the direct analysis method.
    """
    if H <= 0.0 or L <= 0.0 or drift <= 0.0:
        raise GeometryError(f"H, L and drift must be positive, got {H}, {L}, {drift}")
    return RM * H * L / drift


def B2_multiplier(Pstory: Kip, Pe_story: Kip, basis: Basis) -> Dimensionless:
    """P-Delta amplifier for one story and one direction of translation.

    AISC 360-16, Eq. A-8-6, Sect. 8.2.2, p. 16.1-252::

        B2 = 1/(1 - alpha*Pstory/Pe,story)  >= 1

    ``Pstory`` is the **total** vertical load on the story, "including loads in
    columns that are not part of the lateral force-resisting system" -- leaning
    columns push on the frame whether or not they resist lateral load, and
    omitting them understates ``B2``.

    ``B2`` doubles as the second-order/first-order drift ratio that Appendix 7
    uses to test whether the effective length and first-order analysis methods
    are permitted (both require it at or below 1.5).

    Raises
    ------
    OutOfScopeError
        If ``alpha*Pstory >= Pe,story``. The story has reached its sidesway
        buckling load and the frame is unstable, not merely amplified.
    """
    if Pe_story <= 0.0:
        raise GeometryError(f"Pe,story must be positive, got {Pe_story}")
    if Pstory <= 0.0:
        return 1.0

    demand = alpha_for(basis) * Pstory / Pe_story
    if demand >= 1.0:
        raise OutOfScopeError(
            f"alpha*Pstory/Pe,story = {demand:.4f} reaches 1.0: the story is at or "
            "beyond its sidesway buckling load. The frame is unstable -- stiffen it."
        )
    return max(1.0 / (1.0 - demand), 1.0)


# ===========================================================================
# Sect. 8.2 -- assembling the amplified forces
# ===========================================================================
@dataclass(frozen=True, slots=True)
class AmplifiedForces:
    """Second-order required strengths from Eqs. A-8-1 and A-8-2."""

    Mr: KipIn
    Pr: Kip
    B1: Dimensionless
    B2: Dimensionless

    def report(self, width: int = 78) -> str:
        lines = ["=" * width, "Appendix 8 -- approximate second-order analysis", "=" * width]
        lines.append(f"{'B1 (P-delta, member)':<58}{self.B1:>20,.4f}")
        lines.append(f"{'B2 (P-Delta, story)':<58}{self.B2:>20,.4f}")
        lines.append("-" * width)
        lines.append(f"{'Mr = B1*Mnt + B2*Mlt   (Eq. A-8-1)':<58}{self.Mr:>20,.4g}")
        lines.append(f"{'Pr = Pnt + B2*Plt      (Eq. A-8-2)':<58}{self.Pr:>20,.4g}")
        lines.append("=" * width)
        return "\n".join(lines)


def amplified_forces(
    Mnt: KipIn, Mlt: KipIn, Pnt: Kip, Plt: Kip, B1: Dimensionless, B2: Dimensionless
) -> AmplifiedForces:
    """Required second-order flexural and axial strengths.

    AISC 360-16, Eqs. A-8-1 and A-8-2, Sect. 8.2, p. 16.1-249::

        Mr = B1*Mnt + B2*Mlt
        Pr = Pnt   + B2*Plt

    Note the asymmetry: ``B1`` appears only on ``Mnt``, and ``Pnt`` is not
    amplified at all. ``B1`` is a moment effect -- the member's own curvature
    does not change the axial force it carries -- so there is no ``B1*Pnt`` term.

    The User Note on p. 16.1-250 records that these equations apply to all
    members in all structures, but that ``B1`` other than unity applies only to
    beam-column moments while ``B2`` applies to moments *and* axial forces in
    components of the lateral force-resisting system.
    """
    if B1 < 1.0 or B2 < 1.0:
        raise AISC360Error(
            f"B1 and B2 are amplifiers and cannot fall below 1.0, got B1={B1}, B2={B2}"
        )
    return AmplifiedForces(
        Mr=B1 * Mnt + B2 * Mlt,  # Eq. A-8-1
        Pr=Pnt + B2 * Plt,  # Eq. A-8-2
        B1=B1, B2=B2,
    )
