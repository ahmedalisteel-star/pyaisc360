"""Chapter D -- Design of Members for Tension.

Covers Sects. D1 through D6, pp. 16.1-26 to 16.1-32.

The chapter is short but carries one trap the rest of the Specification does
not: **its two limit states have different resistance factors**. Sect. D2 gives
tensile yielding ``phi = 0.90 / Omega = 1.67`` and tensile rupture
``phi = 0.75 / Omega = 2.00``. So the two cannot be compared on nominal strength
-- the governing state must be decided on *available* strength, and the two
orderings genuinely disagree when::

    0.833 < Fy*Ag / (Fu*Ae) < 1.0

Inside that band rupture has the larger ``Pn`` but the smaller ``phi*Pn``, and
it is ``phi*Pn`` that has to satisfy Eq. B3-1. :class:`~pyaisc360.StrengthResult`
carries per-limit-state factors for exactly this reason.

The higher factor on yielding is deliberate: yielding of the gross section is a
ductile, self-limiting failure that redistributes load, while rupture of the net
section is sudden. The Specification insures more heavily against the brittle one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .core.citations import cite
from .core.config import Basis
from .core.enums import LimitState
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.result import LimitStateResult, StrengthResult
from .core.units import Inch, Inch2, Kip, Ksi, Ratio
from .materials import Steel
from .sections import require_properties

__all__ = [
    "PHI_T_YIELDING",
    "OMEGA_T_YIELDING",
    "PHI_T_RUPTURE",
    "OMEGA_T_RUPTURE",
    "SLENDERNESS_ADVISORY_LIMIT",
    "ShearLagCase",
    "TensionMember",
    "tensile_yielding",
    "tensile_rupture",
    "effective_net_area",
    "shear_lag_factor",
    "shear_lag_case_2",
    "shear_lag_case_4",
    "shear_lag_case_5",
    "shear_lag_case_6",
    "pin_tensile_rupture",
    "pin_shear_rupture",
    "pin_effective_edge_distance",
    "check_pin_dimensional_requirements",
    "tensile_strength",
]

#: Sect. D2(a), p. 16.1-28 -- tensile yielding in the gross section.
PHI_T_YIELDING: float = 0.90
OMEGA_T_YIELDING: float = 1.67

#: Sect. D2(b), p. 16.1-28 -- tensile rupture in the net section. Also applies
#: to the pin-connected limit states of Sect. D5.1.
PHI_T_RUPTURE: float = 0.75
OMEGA_T_RUPTURE: float = 2.00

#: Sect. D1 User Note, p. 16.1-26: L/r "preferably should not exceed 300".
#: Sect. D1 opens by stating there is **no maximum slenderness limit** for
#: tension members, so this is advisory only -- and the User Note excludes rods
#: and hangers even from the suggestion.
SLENDERNESS_ADVISORY_LIMIT: Ratio = 300.0

#: Relative slack on Table D3.1's geometric thresholds (l >= 1.3D, l >= H).
#: Products such as ``1.3 * 6.0`` land one ULP above the exact value, which
#: would drop an exactly-conforming connection off its plateau.
_BOUNDARY_TOL: float = 1e-12


class ShearLagCase(str, Enum):
    """The eight cases of Table D3.1, pp. 16.1-30 to 16.1-31."""

    #: Case 1 -- load transmitted directly to each cross-sectional element.
    #: U = 1.0.
    ALL_ELEMENTS_CONNECTED = "all elements connected directly"
    #: Case 2 -- load transmitted to some but not all elements. U = 1 - xbar/l.
    #: The general case, and always permitted as an alternative to 7 and 8.
    SOME_ELEMENTS_CONNECTED = "some elements connected (general case)"
    #: Case 3 -- transverse welds only. U = 1.0, but An is the area of the
    #: *directly connected* elements alone.
    TRANSVERSE_WELDS_ONLY = "transverse welds only"
    #: Case 4 -- longitudinal welds only, on plates and similar.
    LONGITUDINAL_WELDS_ONLY = "longitudinal welds only"
    #: Case 5 -- round HSS with a single concentric gusset through slots.
    ROUND_HSS_GUSSET = "round HSS with concentric gusset"
    #: Case 6 -- rectangular HSS with gusset plates.
    RECTANGULAR_HSS_GUSSET = "rectangular HSS with gusset"
    #: Case 7 -- W, M, S, HP shapes and tees cut from them, bolted.
    W_SHAPE_BOLTED = "W/M/S/HP shape or tee, bolted"
    #: Case 8 -- single and double angles, bolted.
    ANGLE_BOLTED = "single or double angle, bolted"


@dataclass(frozen=True)
class TensionMember:
    """A member in axial tension.

    Attributes
    ----------
    section:
        Anything satisfying the Wave 0 section contract; ``Ag`` is the minimum.
    steel:
        Grade, supplying ``Fy`` and ``Fu``.
    An:
        Net area, per Sect. B4.3b. Defaults to ``Ag`` for a welded member with
        no holes.
    U:
        Shear lag factor from Table D3.1 -- compute with
        :func:`shear_lag_factor`.
    L_over_r:
        Slenderness, for the Sect. D1 advisory note only. Never affects strength.
    connected_area_ratio:
        Ratio of the gross area of the connected element(s) to the member gross
        area. Sect. D3 states ``U`` "need not be less than" this for open shapes
        -- W, M, S, C, HP, WT, ST and single and double angles. It explicitly
        does **not** apply to closed sections such as HSS, nor to plates.
    """

    section: Any
    steel: Steel
    An: Inch2 | None = None
    U: Ratio = 1.0
    L_over_r: Ratio | None = None
    connected_area_ratio: Ratio | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.U <= 1.0:
            raise GeometryError(f"shear lag factor U must lie in (0, 1], got {self.U}")
        if self.An is not None and self.An <= 0.0:
            raise GeometryError(f"net area must be positive, got {self.An}")


# ===========================================================================
# Sect. D2 -- Tensile Strength
# ===========================================================================
def tensile_yielding(Fy: Ksi, Ag: Inch2) -> Kip:
    """Nominal tensile strength for yielding in the gross section.

    AISC 360-16, Eq. D2-1, Sect. D2(a), p. 16.1-28::

        Pn = Fy*Ag

    with ``phi_t = 0.90`` (LRFD) and ``Omega_t = 1.67`` (ASD).
    """
    if Fy <= 0.0 or Ag <= 0.0:
        raise GeometryError(f"Fy and Ag must be positive, got Fy={Fy}, Ag={Ag}")
    return Fy * Ag


def tensile_rupture(Fu: Ksi, Ae: Inch2) -> Kip:
    """Nominal tensile strength for rupture in the net section.

    AISC 360-16, Eq. D2-2, Sect. D2(b), p. 16.1-28::

        Pn = Fu*Ae

    with ``phi_t = 0.75`` (LRFD) and ``Omega_t = 2.00`` (ASD) -- **not** the
    yielding factors. Sect. D2 also notes that where connections use plug, slot
    or fillet welds in holes or slots, the effective net area *through the holes*
    must be used here.
    """
    if Fu <= 0.0 or Ae <= 0.0:
        raise GeometryError(f"Fu and Ae must be positive, got Fu={Fu}, Ae={Ae}")
    return Fu * Ae


def effective_net_area(An: Inch2, U: Ratio) -> Inch2:
    """Effective net area ``Ae = An*U``.

    AISC 360-16, Eq. D3-1, Sect. D3, p. 16.1-28.
    """
    if An <= 0.0:
        raise GeometryError(f"net area must be positive, got {An}")
    if not 0.0 < U <= 1.0:
        raise GeometryError(f"U must lie in (0, 1], got {U}")
    return An * U


# ===========================================================================
# Table D3.1 -- Shear Lag Factors
# ===========================================================================
def shear_lag_case_2(xbar: Inch, length: Inch) -> Ratio:
    """Case 2 -- the general shear lag expression.

    AISC 360-16, Table D3.1 case 2, p. 16.1-30::

        U = 1 - xbar/l

    ``xbar`` is the connection eccentricity: the distance from the connection
    plane to the centroid of the member (or of the half-section, for a W-shape
    connected by its flanges). ``l`` is the length of the connection.

    Case 2 is always permitted as an alternative to cases 7 and 8, and Table
    D3.1 directs that the **larger** of the two be used.
    """
    if length <= 0.0:
        raise GeometryError(f"connection length must be positive, got {length}")
    if xbar < 0.0:
        raise GeometryError(f"eccentricity must be non-negative, got {xbar}")
    if xbar >= length:
        raise OutOfScopeError(
            f"eccentricity xbar = {xbar} is not less than the connection length "
            f"{length}; U would be non-positive. Lengthen the connection."
        )
    return 1.0 - xbar / length


def shear_lag_case_4(l1: Inch, l2: Inch, w: Inch, xbar: Inch) -> Ratio:
    """Case 4 -- plates and similar connected by longitudinal welds only.

    AISC 360-16, Table D3.1 case 4, p. 16.1-30::

        U = 3*l^2/(3*l^2 + w^2) * (1 - xbar/l),   l = (l1 + l2)/2

    New in the 2016 edition -- it replaced the 2010 table's three-step
    1.0 / 0.87 / 0.75 and, per the Preface, adds "a shear lag factor for welded
    plates or connected elements with unequal length longitudinal welds". The
    averaging of ``l1`` and ``l2`` is what handles the unequal case.

    Parameters
    ----------
    l1, l2:
        Lengths of the two longitudinal welds, in. Equal for a symmetric pair.
    w:
        Width of the plate, in. -- the distance between the welds.
    xbar:
        Connection eccentricity, per case 2. Zero for a plate welded
        symmetrically along both edges.
    """
    if min(l1, l2, w) <= 0.0:
        raise GeometryError(f"l1, l2 and w must be positive, got {l1}, {l2}, {w}")
    length = (l1 + l2) / 2.0
    if xbar >= length:
        raise OutOfScopeError(f"eccentricity {xbar} is not less than mean weld length {length}")
    return 3.0 * length**2 / (3.0 * length**2 + w**2) * (1.0 - xbar / length)


def shear_lag_case_5(D: Inch, length: Inch) -> Ratio:
    """Case 5 -- round HSS with a single concentric gusset through slots.

    AISC 360-16, Table D3.1 case 5, p. 16.1-31::

        l >= 1.3*D:        U = 1.0
        D <= l < 1.3*D:    U = 1 - xbar/l,  with xbar = D/pi

    ``xbar = D/pi`` is the centroid of a half-circumference -- the geometric
    result for a tube pulling against a diametral plate.

    Raises
    ------
    OutOfScopeError
        If ``l < D``. Table D3.1 gives no factor there; the slot is too short
        for the provision.
    """
    if D <= 0.0 or length <= 0.0:
        raise GeometryError(f"D and l must be positive, got {D}, {length}")
    # Compared with a relative tolerance: 1.3*6.0 evaluates to 7.800000000000001,
    # so an exact l = 1.3*D would otherwise miss the plateau by one ULP and
    # return 0.755 instead of 1.0. Table D3.1's thresholds are design values, and
    # a part in 1e-12 has no physical meaning at either side of them.
    if length >= 1.3 * D * (1.0 - _BOUNDARY_TOL):
        return 1.0
    if length < D * (1.0 - _BOUNDARY_TOL):
        raise OutOfScopeError(
            f"connection length {length} is below D = {D}. Table D3.1 case 5 "
            "covers l >= D only."
        )
    return 1.0 - (D / math.pi) / length


def shear_lag_case_6(B: Inch, H: Inch, length: Inch, *, two_side_plates: bool = False) -> Ratio:
    """Case 6 -- rectangular HSS with gusset plates.

    AISC 360-16, Table D3.1 case 6, p. 16.1-31, both for ``l >= H``::

        single concentric gusset:  U = 1 - xbar/l,  xbar = (B^2 + 2*B*H)/(4*(B+H))
        two side gusset plates:    U = 1 - xbar/l,  xbar = B^2/(4*(B+H))

    ``B`` is measured 90 degrees to the plane of the connection and ``H`` in the
    plane of it -- getting them the wrong way round changes ``xbar`` materially.
    The two-plate eccentricity is always the smaller, which is why side plates
    give the better factor.
    """
    if min(B, H, length) <= 0.0:
        raise GeometryError(f"B, H and l must be positive, got {B}, {H}, {length}")
    if length < H * (1.0 - _BOUNDARY_TOL):
        raise OutOfScopeError(
            f"connection length {length} is below H = {H}. Table D3.1 case 6 "
            "covers l >= H only."
        )
    xbar = B**2 / (4.0 * (B + H)) if two_side_plates else (B**2 + 2.0 * B * H) / (4.0 * (B + H))
    return 1.0 - xbar / length


def shear_lag_factor(
    case: ShearLagCase,
    *,
    xbar: Inch | None = None,
    length: Inch | None = None,
    l1: Inch | None = None,
    l2: Inch | None = None,
    w: Inch | None = None,
    D: Inch | None = None,
    B: Inch | None = None,
    H: Inch | None = None,
    two_side_plates: bool = False,
    bf: Inch | None = None,
    d: Inch | None = None,
    fasteners_per_line: int | None = None,
    web_connected: bool = False,
    connected_area_ratio: Ratio | None = None,
) -> tuple[Ratio, str]:
    """Shear lag factor ``U`` from Table D3.1, pp. 16.1-30 to 16.1-31.

    Returns ``(U, explanation)``.

    Cases 7 and 8 are tabulated shortcuts; Table D3.1 permits case 2 instead and
    directs that the **larger** value be used, so both are computed when the
    case-2 inputs are supplied and the larger is returned.

    ``connected_area_ratio`` applies the Sect. D3 floor: for open cross sections
    -- W, M, S, C, HP, WT, ST, single and double angles -- ``U`` "need not be
    less than the ratio of the gross area of the connected element(s) to the
    member gross area". This does **not** apply to HSS or to plates, so it is an
    explicit argument rather than something inferred.
    """
    explanation: str

    if case is ShearLagCase.ALL_ELEMENTS_CONNECTED:
        U, explanation = 1.0, "case 1: load transmitted directly to every element"

    elif case is ShearLagCase.TRANSVERSE_WELDS_ONLY:
        U = 1.0
        explanation = (
            "case 3: U = 1.0, but An must be the area of the DIRECTLY CONNECTED "
            "elements only, not the whole net section"
        )

    elif case is ShearLagCase.SOME_ELEMENTS_CONNECTED:
        if xbar is None or length is None:
            raise AISC360Error("Table D3.1 case 2 needs xbar and length")
        U = shear_lag_case_2(xbar, length)
        explanation = f"case 2: U = 1 - {xbar:g}/{length:g}"

    elif case is ShearLagCase.LONGITUDINAL_WELDS_ONLY:
        if l1 is None or l2 is None or w is None:
            raise AISC360Error("Table D3.1 case 4 needs l1, l2 and w")
        U = shear_lag_case_4(l1, l2, w, xbar or 0.0)
        explanation = f"case 4: longitudinal welds, l = ({l1:g} + {l2:g})/2"

    elif case is ShearLagCase.ROUND_HSS_GUSSET:
        if D is None or length is None:
            raise AISC360Error("Table D3.1 case 5 needs D and length")
        U = shear_lag_case_5(D, length)
        explanation = f"case 5: round HSS, l/D = {length / D:.2f}"

    elif case is ShearLagCase.RECTANGULAR_HSS_GUSSET:
        if B is None or H is None or length is None:
            raise AISC360Error("Table D3.1 case 6 needs B, H and length")
        U = shear_lag_case_6(B, H, length, two_side_plates=two_side_plates)
        explanation = (
            f"case 6: rectangular HSS, {'two side plates' if two_side_plates else 'single gusset'}"
        )

    elif case is ShearLagCase.W_SHAPE_BOLTED:
        if web_connected:
            if fasteners_per_line is not None and fasteners_per_line < 4:
                raise OutOfScopeError(
                    "Table D3.1 case 7 web connection requires four or more "
                    f"fasteners per line, got {fasteners_per_line}"
                )
            U, explanation = 0.70, "case 7: web connected, 4+ fasteners per line"
        else:
            if bf is None or d is None:
                raise AISC360Error("Table D3.1 case 7 flange connection needs bf and d")
            if fasteners_per_line is not None and fasteners_per_line < 3:
                raise OutOfScopeError(
                    "Table D3.1 case 7 flange connection requires three or more "
                    f"fasteners per line, got {fasteners_per_line}"
                )
            if bf >= 2.0 / 3.0 * d:
                U, explanation = 0.90, f"case 7: flange connected, bf/d = {bf / d:.3f} >= 2/3"
            else:
                U, explanation = 0.85, f"case 7: flange connected, bf/d = {bf / d:.3f} < 2/3"

    elif case is ShearLagCase.ANGLE_BOLTED:
        if fasteners_per_line is None:
            raise AISC360Error("Table D3.1 case 8 needs fasteners_per_line")
        if fasteners_per_line >= 4:
            U, explanation = 0.80, "case 8: angle, 4+ fasteners per line"
        elif fasteners_per_line == 3:
            U, explanation = 0.60, "case 8: angle, 3 fasteners per line"
        else:
            raise OutOfScopeError(
                f"Table D3.1 case 8 covers three or more fasteners per line, got "
                f"{fasteners_per_line}. With fewer than three, use case 2."
            )

    else:  # pragma: no cover - the enum is exhaustive
        raise AISC360Error(f"unhandled shear lag case {case}")

    # Cases 7 and 8 may be superseded by the larger case-2 value.
    if case in {ShearLagCase.W_SHAPE_BOLTED, ShearLagCase.ANGLE_BOLTED} and (
        xbar is not None and length is not None
    ):
        alternative = shear_lag_case_2(xbar, length)
        if alternative > U:
            U = alternative
            explanation += f"; superseded by case 2 (U = {alternative:.3f}, larger)"

    # Sect. D3 floor for open cross sections.
    if connected_area_ratio is not None and connected_area_ratio > U:
        explanation += (
            f"; raised to the Sect. D3 floor, connected/gross area = "
            f"{connected_area_ratio:.3f}"
        )
        U = min(connected_area_ratio, 1.0)

    return U, explanation


# ===========================================================================
# Sect. D5 -- Pin-Connected Members
# ===========================================================================
def pin_effective_edge_distance(t: Inch, actual_edge_distance: Inch) -> Inch:
    """Effective edge distance ``be`` for a pin-connected member.

    AISC 360-16, Sect. D5.1(a), p. 16.1-30::

        be = 2*t + 0.63 in.

    "but not more than the actual distance from the edge of the hole to the edge
    of the part measured in the direction normal to the applied force".
    """
    if t <= 0.0 or actual_edge_distance <= 0.0:
        raise GeometryError(
            f"t and edge distance must be positive, got {t}, {actual_edge_distance}"
        )
    return min(2.0 * t + 0.63, actual_edge_distance)


def pin_tensile_rupture(Fu: Ksi, t: Inch, be: Inch) -> Kip:
    """Tensile rupture on the net effective area of a pin-connected member.

    AISC 360-16, Eq. D5-1, Sect. D5.1(a), p. 16.1-30::

        Pn = Fu*(2*t*be)

    with ``phi_t = 0.75`` / ``Omega_t = 2.00``. The factor of two is the two
    sides of the pin hole.
    """
    if min(Fu, t, be) <= 0.0:
        raise GeometryError(f"Fu, t and be must be positive, got {Fu}, {t}, {be}")
    return Fu * (2.0 * t * be)


def pin_shear_rupture(Fu: Ksi, t: Inch, a: Inch, d: Inch) -> Kip:
    """Shear rupture on the effective area of a pin-connected member.

    AISC 360-16, Eq. D5-2, Sect. D5.1(b), p. 16.1-30::

        Pn = 0.6*Fu*Asf,   Asf = 2*t*(a + d/2)

    with ``phi_sf = 0.75`` / ``Omega_sf = 2.00``.

    Parameters
    ----------
    a:
        Shortest distance from the edge of the pin hole to the edge of the
        member, measured **parallel to the direction of the force**.
    d:
        Diameter of the pin.
    """
    if min(Fu, t, a, d) <= 0.0:
        raise GeometryError(f"Fu, t, a and d must be positive, got {Fu}, {t}, {a}, {d}")
    Asf = 2.0 * t * (a + d / 2.0)
    return 0.6 * Fu * Asf


def check_pin_dimensional_requirements(
    plate_width: Inch, be: Inch, d: Inch, a: Inch, *, strict: bool = True
) -> list[str]:
    """Sect. D5.2 dimensional requirements, p. 16.1-30.

    Returns the list of violations; raises when ``strict`` and any are found::

        (c) plate width at the pin hole >= 2*be + d
        (c) extension a beyond the bearing end >= 1.33*be

    Requirements (a), (b) and (d) -- hole centred on the member, hole no more
    than 1/32 in. larger than the pin, and the 45-degree corner cut rule -- are
    detailing facts this function cannot see.
    """
    violations: list[str] = []
    if plate_width < 2.0 * be + d:
        violations.append(
            f"Sect. D5.2(c): plate width {plate_width:g} is below 2*be + d = {2 * be + d:g}"
        )
    if a < 1.33 * be:
        violations.append(
            f"Sect. D5.2(c): extension a = {a:g} is below 1.33*be = {1.33 * be:g}"
        )
    if strict and violations:
        raise OutOfScopeError("; ".join(violations))
    return violations


# ===========================================================================
# Orchestrator
# ===========================================================================
def tensile_strength(
    member: TensionMember, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Nominal and available tensile strength, ``Pn``, in kips.

    AISC 360-16, Sect. D2, p. 16.1-28: "the lower value obtained according to
    the limit states of tensile yielding in the gross section and tensile rupture
    in the net section".

    "Lower value" means lower **available** strength: the two limit states carry
    different factors (0.90 against 0.75), so comparing nominal strengths would
    pick the wrong one whenever ``0.833 < Fy*Ag/(Fu*Ae) < 1``. The factors travel
    on the limit states and
    :attr:`~pyaisc360.core.result.StrengthResult.governing` compares on available
    strength.
    """
    steel = member.steel
    Ag = require_properties(member.section, "Ag")["Ag"]
    An = member.An if member.An is not None else Ag
    if An > Ag:
        raise GeometryError(f"net area {An} exceeds gross area {Ag}")

    Ae = effective_net_area(An, member.U)  # Eq. D3-1

    states = [
        LimitStateResult(
            LimitState.TENSILE_YIELDING,
            tensile_yielding(steel.Fy, Ag),  # Eq. D2-1
            cite("D2-1"),
            detail={"Fy": steel.Fy, "Ag": Ag},
            note="yielding on the gross section -- ductile, so phi = 0.90",
            phi=PHI_T_YIELDING, omega=OMEGA_T_YIELDING,
        ),
        LimitStateResult(
            LimitState.TENSILE_RUPTURE,
            tensile_rupture(steel.Fu, Ae),  # Eq. D2-2
            cite("D2-2"),
            detail={"Fu": steel.Fu, "An": An, "U": member.U, "Ae": Ae},
            note="rupture on the net section -- sudden, so phi = 0.75",
            phi=PHI_T_RUPTURE, omega=OMEGA_T_RUPTURE,
        ),
    ]

    result = StrengthResult.build(
        "Pn", "kip", states,
        phi=PHI_T_YIELDING, omega=OMEGA_T_YIELDING, basis=basis,
    )

    if member.L_over_r is not None and member.L_over_r > SLENDERNESS_ADVISORY_LIMIT:
        governing = result.governing
        annotated = LimitStateResult(
            governing.limit_state, governing.nominal, governing.citation,
            detail=governing.detail,
            note=(
                f"{governing.note}; L/r = {member.L_over_r:.0f} exceeds the 300 the "
                "Sect. D1 User Note suggests -- advisory only, Sect. D1 states there "
                "is no maximum slenderness for tension members"
            ),
            phi=governing.phi, omega=governing.omega,
        )
        result = StrengthResult(
            symbol=result.symbol, unit=result.unit, phi=result.phi, omega=result.omega,
            basis=result.basis,
            limit_states=tuple(
                annotated if ls is governing else ls for ls in result.limit_states
            ),
        )
    return result
