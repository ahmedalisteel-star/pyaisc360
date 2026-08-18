"""Appendix 1 -- Design by Advanced Analysis.

Covers Sects. 1.1 through 1.3, pp. 16.1-185 to 16.1-191.

Appendix 1 extends the direct analysis method of Chapter C by moving more of
the design into the analysis:

* **Sect. 1.2** adds *member* imperfections -- initial out-of-straightness -- to
  the *system* imperfections Chapter C already covers. Sect. C2.2's User Note
  is explicit that member out-of-straightness "is accounted for in the
  compression member design provisions of Chapter E and need not be considered
  explicitly in the analysis"; modelling it directly is what Sect. 1.2 permits
  instead.
* **Sect. 1.3** goes further and allows *inelastic* analysis, which is where the
  numbered equations live. Modelling redistribution means the sections forming
  plastic hinges have to be able to rotate, so Sect. 1.3.2 imposes tighter
  slenderness and bracing limits than Chapter B and F do.

Every equation here is therefore a **ductility** limit, not a strength one:
``lambda_pd`` in place of Table B4.1b's ``lambda_p``, and ``Lpd`` in place of
Eq. F2-5's ``Lp``. They are the price of the extra capacity inelastic analysis
buys.

Sect. 1.3.1 restricts the whole approach to LRFD and to ``Fy <= 65`` ksi.
"""

from __future__ import annotations

import math

from .core.exceptions import GeometryError
from .core.units import Inch, Kip, KipIn, Ksi, Ratio

__all__ = [
    "PHI_C_INELASTIC",
    "FY_MAX_INELASTIC",
    "lambda_pd_web",
    "lambda_pd_hss_flange",
    "lambda_pd_round_hss",
    "effective_end_moment",
    "Lpd_I_shape",
    "Lpd_bar_or_box",
    "compression_bracing_limit",
    "check_inelastic_limitations",
]

#: Sect. 1.3.2a, p. 16.1-188 -- the resistance factor Eqs. A-1-1 and A-1-2 use.
PHI_C_INELASTIC: float = 0.90

#: Sect. 1.3.1, p. 16.1-187 -- inelastic analysis is limited to this yield stress.
FY_MAX_INELASTIC: Ksi = 65.0


def lambda_pd_web(Pu: Kip, Py: Kip, E: Ksi, Fy: Ksi) -> tuple[Ratio, str]:
    """Ductility slenderness limit for a web under combined flexure and compression.

    AISC 360-16, Eqs. A-1-1 and A-1-2, Sect. 1.3.2b, p. 16.1-188::

        Pu/(phi_c*Py) <= 0.125:
            lambda_pd = 3.76*sqrt(E/Fy)*(1 - 2.75*Pu/(phi_c*Py))       (A-1-1)
        Pu/(phi_c*Py) >  0.125:
            lambda_pd = 1.12*sqrt(E/Fy)*(2.33 - Pu/(phi_c*Py))
                        >= 1.49*sqrt(E/Fy)                              (A-1-2)

    with ``phi_c = 0.90`` and ``Py = Fy*Ag``.

    At zero axial load Eq. A-1-1 gives ``3.76*sqrt(E/Fy)`` -- exactly Table
    B4.1b case 15's ``lambda_p``. Axial load then tightens it: a web carrying
    compression as well as flexure has less rotation capacity, so it must be
    stockier to form a hinge.

    The two branches meet at ``Pu/(phi_c*Py) = 0.125``, where Eq. A-1-1 gives
    ``3.76*(1 - 0.34375) = 2.4664`` and Eq. A-1-2 gives
    ``1.12*(2.33 - 0.125) = 2.4696`` -- a 0.13% step, the Specification's own
    rounding rather than a discontinuity in intent.

    The ``>= 1.49*sqrt(E/Fy)`` floor on Eq. A-1-2 is Table B4.1a case 5's
    **axial** ``lambda_r``: however much axial load the member carries, the web
    need never be stockier than the nonslender limit for pure compression.
    """
    if Py <= 0.0:
        raise GeometryError(f"Py must be positive, got {Py}")
    if Fy <= 0.0 or E <= 0.0:
        raise GeometryError(f"E and Fy must be positive, got {E}, {Fy}")

    root = math.sqrt(E / Fy)
    ratio = abs(Pu) / (PHI_C_INELASTIC * Py)

    if ratio <= 0.125:
        return (
            3.76 * root * (1.0 - 2.75 * ratio),  # Eq. A-1-1
            f"Eq. A-1-1: Pu/(phi_c*Py) = {ratio:.4f} <= 0.125",
        )
    limit = 1.12 * root * (2.33 - ratio)  # Eq. A-1-2
    floor = 1.49 * root
    if limit < floor:
        return floor, (
            f"Eq. A-1-2 floor: 1.49*sqrt(E/Fy) governs at Pu/(phi_c*Py) = {ratio:.4f} "
            "-- the same limit as Table B4.1a case 5 for pure compression"
        )
    return limit, f"Eq. A-1-2: Pu/(phi_c*Py) = {ratio:.4f} > 0.125"


def lambda_pd_hss_flange(E: Ksi, Fy: Ksi) -> Ratio:
    """Ductility slenderness limit for rectangular HSS and box flanges.

    AISC 360-16, Eq. A-1-3, Sect. 1.3.2b, p. 16.1-189::

        lambda_pd = 0.94*sqrt(E/Fy)

    Also covers flange cover plates and diaphragm plates between lines of
    fasteners or welds. Tighter than Table B4.1b case 17's ``lambda_p`` of
    ``1.12*sqrt(E/Fy)`` -- a 16% reduction bought by the rotation demand.
    """
    if E <= 0.0 or Fy <= 0.0:
        raise GeometryError(f"E and Fy must be positive, got {E}, {Fy}")
    return 0.94 * math.sqrt(E / Fy)


def lambda_pd_round_hss(E: Ksi, Fy: Ksi) -> Ratio:
    """Ductility slenderness limit for round HSS in flexure.

    AISC 360-16, Eq. A-1-4, Sect. 1.3.2b, p. 16.1-189::

        lambda_pd = 0.045*E/Fy

    A plain ``E/Fy`` ratio with **no square root** -- the same form as Table
    B4.1b case 20 and Table B4.1a case 9, and tighter than case 20's
    ``0.07*E/Fy``.
    """
    if E <= 0.0 or Fy <= 0.0:
        raise GeometryError(f"E and Fy must be positive, got {E}, {Fy}")
    return 0.045 * E / Fy


def effective_end_moment(M1: KipIn, M2: KipIn, Mmid: KipIn) -> tuple[KipIn, str]:
    """Effective end moment ``M1'`` for the Sect. 1.3.2c unbraced length.

    AISC 360-16, Eqs. A-1-6a, A-1-6b and A-1-6c, Sect. 1.3.2c, p. 16.1-189::

        moment anywhere in the segment exceeds M2:  M1'/M2 = +1     (A-1-6a)
        Mmid <= (M1 + M2)/2:                        M1' = M1        (A-1-6b)
        Mmid >  (M1 + M2)/2:                        M1' = 2*Mmid - M2 < M2
                                                                    (A-1-6c)

    ``M2`` is the larger end moment and is **always positive**. ``M1`` and
    ``Mmid`` are positive when they cause compression in the same flange as
    ``M2``, negative otherwise -- so the sign convention is about which flange
    is compressed, not about sagging or hogging.

    Eq. A-1-6c is the interesting one: a mid-span moment above the average of
    the ends means the moment diagram bulges, and the effective end moment is
    replaced by the value that would give the same bulge linearly. It is capped
    below ``M2``, so a segment with a large mid-span moment behaves as if it
    were in near-uniform moment -- the worst case for lateral-torsional
    buckling, and correctly so.

    Detecting whether a moment "anywhere within the unbraced length exceeds M2"
    needs the full diagram, which this function does not have; pass
    ``Mmid > M2`` or call with :func:`Lpd_I_shape`'s ``exceeds_M2`` flag.
    """
    if M2 <= 0.0:
        raise GeometryError(f"M2 is the larger end moment and must be positive, got {M2}")

    average = (M1 + M2) / 2.0
    if Mmid <= average:
        return M1, "Eq. A-1-6b: Mmid <= (M1 + M2)/2, M1' = M1"
    M1_prime = 2.0 * Mmid - M2
    if M1_prime >= M2:
        return M2, (
            "Eq. A-1-6c capped: 2*Mmid - M2 reaches M2, so the segment is treated as "
            "near-uniform moment -- the worst case for LTB"
        )
    return M1_prime, "Eq. A-1-6c: Mmid > (M1 + M2)/2, M1' = 2*Mmid - M2"


def Lpd_I_shape(M1_prime: KipIn, M2: KipIn, ry: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """Maximum unbraced length at a plastic hinge, I-shape about the major axis.

    AISC 360-16, Eq. A-1-5, Sect. 1.3.2c, p. 16.1-189::

        Lpd = [0.12 - 0.076*(M1'/M2)]*(E/Fy)*ry

    Compare Eq. F2-5's ``Lp = 1.76*ry*sqrt(E/Fy)``: this one uses ``E/Fy``
    **unsquare-rooted**, so it scales differently with grade, and it carries a
    moment-gradient term that ``Lp`` does not.

    At uniform moment (``M1'/M2 = +1``) the bracket is 0.044; at full reverse
    curvature (``-1``) it is 0.196 -- a 4.5-fold range. A hinge in near-uniform
    moment needs bracing more than four times as close.

    The sign convention is Eq. F13-8's and Eq. A-8-4's: positive for reverse
    curvature.
    """
    if M2 <= 0.0:
        raise GeometryError(f"M2 must be positive, got {M2}")
    if ry <= 0.0:
        raise GeometryError(f"ry must be positive, got {ry}")
    ratio = M1_prime / M2
    if not -1.0 <= ratio <= 1.0:
        raise GeometryError(f"M1'/M2 = {ratio} is outside [-1, 1]")
    return (0.12 - 0.076 * ratio) * (E / Fy) * ry


def Lpd_bar_or_box(M1_prime: KipIn, M2: KipIn, ry: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """Maximum unbraced length at a plastic hinge, solid bar or box section.

    AISC 360-16, Eq. A-1-7, Sect. 1.3.2c, p. 16.1-190::

        Lpd = [0.17 - 0.10*(M1'/M2)]*(E/Fy)*ry  >=  0.10*(E/Fy)*ry

    Both the intercept and the gradient term are larger than Eq. A-1-5's, and
    there is a floor Eq. A-1-5 does not have. A closed or solid section has far
    more torsional stiffness, so it tolerates much wider hinge bracing -- and
    the floor stops the uniform-moment case from becoming needlessly tight.
    """
    if M2 <= 0.0:
        raise GeometryError(f"M2 must be positive, got {M2}")
    if ry <= 0.0:
        raise GeometryError(f"ry must be positive, got {ry}")
    ratio = M1_prime / M2
    if not -1.0 <= ratio <= 1.0:
        raise GeometryError(f"M1'/M2 = {ratio} is outside [-1, 1]")
    base = (E / Fy) * ry
    return max((0.17 - 0.10 * ratio) * base, 0.10 * base)


def compression_bracing_limit(r: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """Maximum unbraced length for a compression member containing a plastic hinge.

    AISC 360-16, Sect. 1.3.2c, p. 16.1-190: "For all types of members subject to
    axial compression and containing plastic hinges, the laterally unbraced
    lengths about the cross-section major and minor axes shall not exceed
    ``4.71*rx*sqrt(E/Fy)`` and ``4.71*ry*sqrt(E/Fy)``, respectively."

    The same 4.71 as Sect. E3's elastic/inelastic transition -- a member with a
    hinge must be on the inelastic side of that boundary, because an elastically
    buckling column cannot form one.

    Stated in prose rather than as a numbered equation, which is why it has no
    label.
    """
    if r <= 0.0:
        raise GeometryError(f"the radius of gyration must be positive, got {r}")
    return 4.71 * r * math.sqrt(E / Fy)


def check_inelastic_limitations(Fy: Ksi, *, lrfd: bool = True) -> list[str]:
    """Sect. 1.3.1 limitations on design by inelastic analysis, p. 16.1-187.

    Returns the list of violations::

        Fy <= 65 ksi
        LRFD only

    Sect. 1.3.1 also requires continuous lateral-torsional bracing detailing,
    that members be compact, and that the structure's stability be assessed by
    the direct analysis method -- conditions this function cannot see.
    """
    violations: list[str] = []
    if Fy > FY_MAX_INELASTIC:
        violations.append(
            f"Sect. 1.3.1: Fy = {Fy:g} ksi exceeds {FY_MAX_INELASTIC:g} ksi -- higher "
            "grades lack the rotation capacity inelastic analysis assumes"
        )
    if not lrfd:
        violations.append(
            "Sect. 1.3.1: design by inelastic analysis is permitted under LRFD only"
        )
    return violations
