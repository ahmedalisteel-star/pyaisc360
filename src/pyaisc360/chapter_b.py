"""Chapter B -- Member Properties: the width-to-thickness limit tables.

Table B4.1a (axial compression, p. 16.1-16) feeds Chapter E; Table B4.1b
(flexure, pp. 16.1-18 to 16.1-19) feeds Chapter F.

Both are tables of coefficients, not of numbers -- nearly every entry is
``coefficient * sqrt(E/Fy)``. Encoding them as coefficients rather than as
evaluated limits keeps one implementation of the square root
(:func:`pyaisc360.utils.limiting_ratio`) and makes the grade dependence
explicit. The exceptions are the round-HSS rows, which are plain ``E/Fy``
ratios with no square root at all, and the two rows whose limits depend on
another property (``kc``, ``FL``, ``Mp/My``).
"""

from __future__ import annotations

import math
from enum import Enum

from .core.enums import AxialSlenderness, FlexuralSlenderness
from .core.exceptions import AISC360Error, GeometryError
from .core.units import Ksi, Ratio
from .utils import classify_axial_element, classify_flexural_element, limiting_ratio

__all__ = [
    "AxialElement",
    "FlexuralElement",
    "kc_coefficient",
    "limiting_width_to_thickness",
    "classify_axial",
    "flexural_limits",
    "classify_flexural",
]


class AxialElement(str, Enum):
    """The nine cases of Table B4.1a, p. 16.1-16.

    Naming follows the table's own "Description of Element" column. The case
    number is carried so a report can cite it.
    """

    #: Case 1 -- flanges of rolled I-shapes, plates projecting from rolled
    #: I-shapes, outstanding legs of pairs of angles in continuous contact,
    #: flanges of channels, and flanges of tees. b/t, 0.56*sqrt(E/Fy).
    ROLLED_I_FLANGE = "rolled I-shape flange"
    #: Case 2 -- flanges of built-up I-shapes and plates or angle legs
    #: projecting from them. b/t, 0.64*sqrt(kc*E/Fy).
    BUILT_UP_I_FLANGE = "built-up I-shape flange"
    #: Case 3 -- legs of single angles, legs of double angles with separators,
    #: and all other unstiffened elements. b/t, 0.45*sqrt(E/Fy).
    ANGLE_LEG = "angle leg or other unstiffened element"
    #: Case 4 -- stems of tees. d/t, 0.75*sqrt(E/Fy).
    TEE_STEM = "tee stem"
    #: Case 5 -- webs of doubly symmetric rolled and built-up I-shapes and
    #: channels. h/tw, 1.49*sqrt(E/Fy).
    I_WEB = "I-shape or channel web"
    #: Case 6 -- walls of rectangular HSS. b/t, 1.40*sqrt(E/Fy).
    RECTANGULAR_HSS_WALL = "rectangular HSS wall"
    #: Case 7 -- flange cover plates and diaphragm plates between lines of
    #: fasteners or welds. b/t, 1.40*sqrt(E/Fy).
    COVER_PLATE = "flange cover plate or diaphragm plate"
    #: Case 8 -- all other stiffened elements. b/t, 1.49*sqrt(E/Fy).
    OTHER_STIFFENED = "other stiffened element"
    #: Case 9 -- round HSS. D/t, 0.11*E/Fy -- note: no square root.
    ROUND_HSS = "round HSS"

    @property
    def case(self) -> int:
        return _CASE_NUMBER[self]

    @property
    def is_stiffened(self) -> bool:
        """Supported along two edges. Drives the Table E7.1 factor selection."""
        return self in {
            AxialElement.I_WEB,
            AxialElement.RECTANGULAR_HSS_WALL,
            AxialElement.COVER_PLATE,
            AxialElement.OTHER_STIFFENED,
        }


_CASE_NUMBER: dict[AxialElement, int] = {
    AxialElement.ROLLED_I_FLANGE: 1,
    AxialElement.BUILT_UP_I_FLANGE: 2,
    AxialElement.ANGLE_LEG: 3,
    AxialElement.TEE_STEM: 4,
    AxialElement.I_WEB: 5,
    AxialElement.RECTANGULAR_HSS_WALL: 6,
    AxialElement.COVER_PLATE: 7,
    AxialElement.OTHER_STIFFENED: 8,
    AxialElement.ROUND_HSS: 9,
}

#: Coefficient multiplying ``sqrt(E/Fy)`` in Table B4.1a. Case 2 is absent
#: because it also carries ``kc``; case 9 is absent because it has no square root.
_COEFFICIENT: dict[AxialElement, float] = {
    AxialElement.ROLLED_I_FLANGE: 0.56,
    AxialElement.ANGLE_LEG: 0.45,
    AxialElement.TEE_STEM: 0.75,
    AxialElement.I_WEB: 1.49,
    AxialElement.RECTANGULAR_HSS_WALL: 1.40,
    AxialElement.COVER_PLATE: 1.40,
    AxialElement.OTHER_STIFFENED: 1.49,
}


def kc_coefficient(h_over_tw: Ratio) -> Ratio:
    """Web-slenderness coefficient ``kc`` for a built-up flange.

    AISC 360-16, Table B4.1a footnote [a], p. 16.1-17::

        kc = 4/sqrt(h/tw),  0.35 <= kc <= 0.76

    The bounds are mandatory ("shall not be taken less than 0.35 nor greater
    than 0.76 for calculation purposes"), not advisory, so they are clamped
    rather than checked.
    """
    if h_over_tw <= 0.0:
        raise GeometryError(f"h/tw must be positive, got {h_over_tw}")
    return min(max(4.0 / math.sqrt(h_over_tw), 0.35), 0.76)


def limiting_width_to_thickness(
    element: AxialElement,
    E: Ksi,
    Fy: Ksi,
    *,
    h_over_tw: Ratio | None = None,
) -> Ratio:
    """Limiting ratio ``lambda_r`` separating nonslender from slender elements.

    AISC 360-16, Table B4.1a, p. 16.1-16.

    Parameters
    ----------
    element:
        Which of the nine table cases applies.
    E, Fy:
        Modulus of elasticity and yield stress, ksi.
    h_over_tw:
        Web slenderness, required for case 2 (built-up I-shape flange) because
        ``lambda_r`` there depends on ``kc``.

    Raises
    ------
    AISC360Error
        If case 2 is requested without ``h_over_tw``.
    """
    if element is AxialElement.ROUND_HSS:
        # Case 9 is the odd one out: 0.11*E/Fy, a plain ratio with no sqrt.
        # Reading it as 0.11*sqrt(E/Fy) understates the limit by a factor of
        # about 22 at Fy = 46 ksi, which would call every HSS slender.
        if Fy <= 0.0 or E <= 0.0:
            raise AISC360Error(f"E and Fy must be positive, got E={E}, Fy={Fy}")
        return 0.11 * E / Fy

    if element is AxialElement.BUILT_UP_I_FLANGE:
        if h_over_tw is None:
            raise AISC360Error(
                "Table B4.1a case 2 (built-up I-shape flange) needs h/tw to "
                "evaluate kc; pass h_over_tw="
            )
        return limiting_ratio(0.64, kc_coefficient(h_over_tw) * E, Fy)

    return limiting_ratio(_COEFFICIENT[element], E, Fy)


def classify_axial(
    element: AxialElement,
    lam: Ratio,
    E: Ksi,
    Fy: Ksi,
    *,
    h_over_tw: Ratio | None = None,
) -> tuple[AxialSlenderness, Ratio]:
    """Classify one element and return ``(classification, lambda_r)``.

    AISC 360-16, Sect. B4.1, p. 16.1-16, with limits from Table B4.1a.

    Returning ``lambda_r`` alongside the verdict saves the caller recomputing
    it; Sect. E7 needs the value itself, not just the comparison.
    """
    lam_r = limiting_width_to_thickness(element, E, Fy, h_over_tw=h_over_tw)
    return classify_axial_element(lam, lam_r), lam_r


# ===========================================================================
# Table B4.1b -- Members Subject to Flexure, pp. 16.1-18 to 16.1-19
# ===========================================================================
class FlexuralElement(str, Enum):
    """The twelve cases of Table B4.1b, numbered 10 through 21 as printed."""

    #: Case 10 -- flanges of rolled I-shapes, channels and tees. 0.38 / 1.0.
    ROLLED_I_FLANGE = "rolled I-shape, channel or tee flange"
    #: Case 11 -- flanges of doubly and singly symmetric built-up I-shapes.
    #: 0.38*sqrt(E/Fy) / 0.95*sqrt(kc*E/FL) -- note the limit uses FL, not Fy.
    BUILT_UP_I_FLANGE = "built-up I-shape flange"
    #: Case 12 -- legs of single angles. 0.54 / 0.91.
    ANGLE_LEG = "single angle leg"
    #: Case 13 -- flanges of I-shapes and channels in minor-axis flexure. 0.38 / 1.0.
    MINOR_AXIS_FLANGE = "I-shape or channel flange, minor-axis flexure"
    #: Case 14 -- stems of tees. 0.84 / 1.52.
    TEE_STEM = "tee stem"
    #: Case 15 -- webs of doubly symmetric I-shapes and channels. 3.76 / 5.70.
    I_WEB = "doubly symmetric I-shape or channel web"
    #: Case 16 -- webs of singly symmetric I-shapes. lambda_p depends on Mp/My.
    SINGLY_SYMMETRIC_I_WEB = "singly symmetric I-shape web"
    #: Case 17 -- flanges of rectangular HSS. 1.12 / 1.40.
    RECTANGULAR_HSS_FLANGE = "rectangular HSS flange"
    #: Case 18 -- flange cover plates and diaphragm plates between lines of
    #: fasteners or welds. 1.12 / 1.40.
    COVER_PLATE = "cover or diaphragm plate"
    #: Case 19 -- webs of rectangular HSS and box sections. 2.42 / 5.70.
    RECTANGULAR_HSS_WEB = "rectangular HSS or box web"
    #: Case 20 -- round HSS. 0.07*E/Fy / 0.31*E/Fy -- no square root.
    ROUND_HSS = "round HSS"
    #: Case 21 -- flanges of box sections. 1.12 / 1.49.
    BOX_FLANGE = "box section flange"

    @property
    def case(self) -> int:
        return _FLEXURAL_CASE[self]


_FLEXURAL_CASE: dict[FlexuralElement, int] = {
    FlexuralElement.ROLLED_I_FLANGE: 10,
    FlexuralElement.BUILT_UP_I_FLANGE: 11,
    FlexuralElement.ANGLE_LEG: 12,
    FlexuralElement.MINOR_AXIS_FLANGE: 13,
    FlexuralElement.TEE_STEM: 14,
    FlexuralElement.I_WEB: 15,
    FlexuralElement.SINGLY_SYMMETRIC_I_WEB: 16,
    FlexuralElement.RECTANGULAR_HSS_FLANGE: 17,
    FlexuralElement.COVER_PLATE: 18,
    FlexuralElement.RECTANGULAR_HSS_WEB: 19,
    FlexuralElement.ROUND_HSS: 20,
    FlexuralElement.BOX_FLANGE: 21,
}

#: (lambda_p coefficient, lambda_r coefficient) on sqrt(E/Fy). Cases 11, 16 and
#: 20 are absent because their limits are not of that form.
_FLEXURAL_COEFFICIENTS: dict[FlexuralElement, tuple[float, float]] = {
    FlexuralElement.ROLLED_I_FLANGE: (0.38, 1.00),
    FlexuralElement.ANGLE_LEG: (0.54, 0.91),
    FlexuralElement.MINOR_AXIS_FLANGE: (0.38, 1.00),
    FlexuralElement.TEE_STEM: (0.84, 1.52),
    FlexuralElement.I_WEB: (3.76, 5.70),
    FlexuralElement.RECTANGULAR_HSS_FLANGE: (1.12, 1.40),
    FlexuralElement.COVER_PLATE: (1.12, 1.40),
    FlexuralElement.RECTANGULAR_HSS_WEB: (2.42, 5.70),
    FlexuralElement.BOX_FLANGE: (1.12, 1.49),
}


def flexural_limits(
    element: FlexuralElement,
    E: Ksi,
    Fy: Ksi,
    *,
    h_over_tw: Ratio | None = None,
    FL: Ksi | None = None,
    Mp_over_My: Ratio | None = None,
    hc_over_hp: Ratio | None = None,
) -> tuple[Ratio, Ratio]:
    """Limiting ratios ``(lambda_p, lambda_r)`` from Table B4.1b.

    AISC 360-16, Table B4.1b, pp. 16.1-18 to 16.1-19.

    Parameters
    ----------
    element:
        Which of the twelve table cases applies.
    E, Fy:
        Modulus of elasticity and yield stress, ksi.
    h_over_tw, FL:
        Required for case 11 (built-up I-shape flange), whose ``lambda_r`` is
        ``0.95*sqrt(kc*E/FL)``. ``FL`` comes from Eq. F4-6a/F4-6b, **not** from
        ``Fy`` -- using ``Fy`` here overstates ``lambda_r`` by up to 20%.
    Mp_over_My, hc_over_hp:
        Required for case 16 (singly symmetric I-shape web), whose
        ``lambda_p`` is::

            (hc/hp)*sqrt(E/Fy) / (0.54*Mp/My - 0.09)^2  <=  lambda_r

    Raises
    ------
    AISC360Error
        If a case-specific argument is missing, or the case-16 denominator is
        non-positive (which happens below ``Mp/My = 0.167`` and is not physical).
    """
    if element is FlexuralElement.ROUND_HSS:
        # Case 20: plain E/Fy ratios, no square root -- as in Table B4.1a case 9.
        if E <= 0.0 or Fy <= 0.0:
            raise AISC360Error(f"E and Fy must be positive, got E={E}, Fy={Fy}")
        return 0.07 * E / Fy, 0.31 * E / Fy

    if element is FlexuralElement.BUILT_UP_I_FLANGE:
        if h_over_tw is None or FL is None:
            raise AISC360Error(
                "Table B4.1b case 11 (built-up I-shape flange) needs h_over_tw "
                "for kc and FL from Eq. F4-6a/F4-6b"
            )
        if FL <= 0.0:
            raise AISC360Error(f"FL must be positive, got {FL}")
        lam_p = limiting_ratio(0.38, E, Fy)
        lam_r = 0.95 * math.sqrt(kc_coefficient(h_over_tw) * E / FL)
        return lam_p, lam_r

    if element is FlexuralElement.SINGLY_SYMMETRIC_I_WEB:
        if Mp_over_My is None or hc_over_hp is None:
            raise AISC360Error(
                "Table B4.1b case 16 (singly symmetric I-shape web) needs "
                "Mp_over_My and hc_over_hp"
            )
        lam_r = limiting_ratio(5.70, E, Fy)
        denominator = 0.54 * Mp_over_My - 0.09
        if denominator <= 0.0:
            raise AISC360Error(
                f"Table B4.1b case 16: (0.54*Mp/My - 0.09) = {denominator:.4g} is "
                f"not positive at Mp/My = {Mp_over_My:.4g}; the shape factor is "
                "below the range the footnote covers"
            )
        lam_p = hc_over_hp * limiting_ratio(1.0, E, Fy) / denominator**2
        # Footnote [c] caps lambda_p at lambda_r.
        return min(lam_p, lam_r), lam_r

    coefficients = _FLEXURAL_COEFFICIENTS[element]
    return (
        limiting_ratio(coefficients[0], E, Fy),
        limiting_ratio(coefficients[1], E, Fy),
    )


def classify_flexural(
    element: FlexuralElement,
    lam: Ratio,
    E: Ksi,
    Fy: Ksi,
    **kwargs: float | None,
) -> tuple[FlexuralSlenderness, Ratio, Ratio]:
    """Classify one element in flexure; return ``(class, lambda_p, lambda_r)``.

    AISC 360-16, Sect. B4.1, p. 16.1-16, with limits from Table B4.1b.

    Chapter F needs the limits themselves as well as the verdict, because the
    noncompact branches interpolate between them.
    """
    lam_p, lam_r = flexural_limits(element, E, Fy, **kwargs)
    return classify_flexural_element(lam, lam_p, lam_r), lam_p, lam_r
