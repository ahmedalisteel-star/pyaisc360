"""Chapter G -- Design of Members for Shear.

Covers Sects. G1 through G7, pp. 16.1-70 to 16.1-76.

Two coefficients do all the work, and telling them apart is the whole chapter:

``Cv1`` -- **web shear strength coefficient**, Eqs. G2-2 to G2-4. Two branches:
    1.0 up to ``1.10*sqrt(kv*E/Fy)``, then an inverse-linear decay. It never
    enters an elastic-buckling branch, because Sect. G2.1 credits the web's
    post-buckling strength implicitly.

``Cv2`` -- **web shear buckling coefficient**, Eqs. G2-9 to G2-11. *Three*
    branches: 1.0, then inverse-linear to ``1.37*sqrt(kv*E/Fy)``, then an
    inverse-square elastic branch. It is the true buckling coefficient, used
    wherever post-buckling strength is not credited -- Sects. G3, G4, G6 and
    inside the tension-field expressions of Sect. G2.2.

Substituting one for the other is unconservative in the range between
``1.10*sqrt(kv*E/Fy)`` and ``1.37*sqrt(kv*E/Fy)``, where they are equal, and
then divergent beyond it. They are separate functions here for that reason.

A note on section numbering: **tension field action is Sect. G2.2 in the 2016
edition**, not Sect. G3. The 2010 edition had it in G3; 360-16 moved it under
Sect. G2 and reassigned G3 to single angles and tees.

Sect. G7 (beams and girders with web openings) states a requirement but no
equations -- it directs the engineer to determine the effect of openings and
reinforce as needed. There is nothing to implement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .core.citations import cite
from .core.config import Basis
from .core.enums import LimitState
from .core.exceptions import AISC360Error, GeometryError
from .core.result import LimitStateResult, StrengthResult
from .core.units import Inch, Inch2, Inch4, Kip, Ksi, Ratio
from .materials import Steel
from .sections import require_properties

__all__ = [
    "PHI_V",
    "OMEGA_V",
    "PHI_V_ROLLED",
    "OMEGA_V_ROLLED",
    "KV_UNSTIFFENED",
    "ShearElement",
    "kv_coefficient",
    "meets_G2_1a",
    "Cv1",
    "Cv2",
    "shear_area_web",
    "g2_1_strength",
    "g2_2_tension_field_strength",
    "tension_field_permitted",
    "transverse_stiffeners_required",
    "stiffener_slenderness_limit",
    "stiffener_moment_of_inertia",
    "g3_strength",
    "g4_strength",
    "g5_critical_stress",
    "g5_strength",
    "g6_strength",
    "shear_strength",
]

#: Sect. G1(a), p. 16.1-70: "For all provisions in this chapter except
#: Section G2.1(a)".
PHI_V: float = 0.90
OMEGA_V: float = 1.67

#: Sect. G2.1(a), p. 16.1-71: the single exception, for webs of **rolled**
#: I-shaped members with ``h/tw <= 2.24*sqrt(E/Fy)``. This is the only place in
#: the Specification where phi = 1.00, and it exists because such a web yields
#: in shear with no buckling at all -- there is no strength reduction to insure
#: against.
PHI_V_ROLLED: float = 1.00
OMEGA_V_ROLLED: float = 1.50

#: Sect. G2.1(b)(2)(i), p. 16.1-71: kv for a web without transverse stiffeners.
#: Also the value Sect. G2.3(a) uses when testing whether stiffeners are needed.
KV_UNSTIFFENED: float = 5.34

#: The Sect. G2.1(a) threshold coefficient.
_G2_1A_LIMIT: float = 2.24


class ShearElement(str, Enum):
    """Which Chapter G section applies to the element carrying the shear."""

    I_WEB = "I-shape or channel web"
    ANGLE_LEG_OR_TEE_STEM = "single angle leg or tee stem"
    RECTANGULAR_HSS = "rectangular HSS or box section"
    ROUND_HSS = "round HSS"
    WEAK_AXIS_FLANGE = "flange in weak-axis shear"


@dataclass(frozen=True)
class ShearMember:
    """A member in shear.

    Attributes
    ----------
    section:
        Anything satisfying the Wave 0 section contract.
    steel:
        Grade, supplying ``Fy`` and ``E``.
    element:
        Which Chapter G section applies.
    h_over_tw:
        Web (or leg, or wall) width-to-thickness ratio. Sect. B4.1b defines what
        ``h`` is for each: clear distance between flanges less the fillet for a
        rolled shape, clear distance for a built-up welded section, distance
        between fastener lines for a built-up bolted section.
    rolled:
        True for a rolled shape. Sect. G2.1(a)'s phi = 1.00 is available only to
        rolled I-shaped members -- a built-up girder with the same ``h/tw`` gets
        phi = 0.90.
    a_over_h:
        Clear stiffener spacing over web depth. ``None`` for an unstiffened web.
    """

    section: Any
    steel: Steel
    element: ShearElement = ShearElement.I_WEB
    h_over_tw: Ratio = 0.0
    rolled: bool = True
    a_over_h: Ratio | None = None

    def __post_init__(self) -> None:
        if self.h_over_tw < 0.0:
            raise GeometryError(f"h/tw must be non-negative, got {self.h_over_tw}")
        if self.a_over_h is not None and self.a_over_h <= 0.0:
            raise GeometryError(f"a/h must be positive, got {self.a_over_h}")


# ===========================================================================
# Shared coefficients
# ===========================================================================
def kv_coefficient(a_over_h: Ratio | None) -> Ratio:
    """Web plate shear buckling coefficient ``kv``.

    AISC 360-16, Sect. G2.1(b)(2) and Eq. G2-5, p. 16.1-71::

        without transverse stiffeners:  kv = 5.34
        with transverse stiffeners:     kv = 5 + 5/(a/h)^2
                                        kv = 5.34 when a/h > 3.0

    The ``a/h > 3.0`` rule is not a smooth continuation: at ``a/h = 3`` the
    formula gives ``5 + 5/9 = 5.556``, and the rule drops it to 5.34. Stiffeners
    spaced further apart than three web depths are treated as if they were not
    there at all, which is the physical reality -- the panel is too long for the
    stiffener to develop a tension field.
    """
    if a_over_h is None:
        return KV_UNSTIFFENED
    if a_over_h <= 0.0:
        raise GeometryError(f"a/h must be positive, got {a_over_h}")
    if a_over_h > 3.0:
        return KV_UNSTIFFENED
    return 5.0 + 5.0 / a_over_h**2  # Eq. G2-5


def meets_G2_1a(h_over_tw: Ratio, E: Ksi, Fy: Ksi, *, rolled: bool = True) -> bool:
    """Whether Sect. G2.1(a)'s ``phi_v = 1.00`` applies.

    AISC 360-16, Sect. G2.1(a), p. 16.1-71: webs of **rolled** I-shaped members
    with ``h/tw <= 2.24*sqrt(E/Fy)``.

    At Fy = 50 ksi the limit is 53.95. The User Note on p. 16.1-71 names the
    ASTM A6 shapes that fail it: W44x230, W40x149, W36x135, W33x118, W30x90,
    W24x55, W16x26 and W12x14 -- everything else qualifies.
    """
    return rolled and h_over_tw <= _G2_1A_LIMIT * math.sqrt(E / Fy)


def Cv1(h_over_tw: Ratio, kv: Ratio, E: Ksi, Fy: Ksi) -> Ratio:
    """Web shear **strength** coefficient.

    AISC 360-16, Eqs. G2-3 and G2-4, Sect. G2.1(b)(1), p. 16.1-71::

        h/tw <= 1.10*sqrt(kv*E/Fy):  Cv1 = 1.0                     (Eq. G2-3)
        h/tw >  1.10*sqrt(kv*E/Fy):  Cv1 = 1.10*sqrt(kv*E/Fy)/(h/tw)  (Eq. G2-4)

    Note there is **no elastic branch**. ``Cv1`` decays as ``1/(h/tw)`` for ever,
    where :func:`Cv2` switches to ``1/(h/tw)^2`` past ``1.37*sqrt(kv*E/Fy)``.
    That difference is Sect. G2.1 crediting post-buckling strength that Sect.
    G2.2's buckling coefficient does not.

    Eq. G2-2 (``Cv1 = 1.0`` for the Sect. G2.1(a) case) is the same value this
    returns whenever the G2.1(a) condition holds, since ``2.24 < 1.10*sqrt(5.34)
    = 2.54``.
    """
    if h_over_tw <= 0.0:
        raise GeometryError(f"h/tw must be positive, got {h_over_tw}")
    limit = 1.10 * math.sqrt(kv * E / Fy)
    if h_over_tw <= limit:
        return 1.0  # Eq. G2-3
    return limit / h_over_tw  # Eq. G2-4


def Cv2(h_over_tw: Ratio, kv: Ratio, E: Ksi, Fy: Ksi) -> Ratio:
    """Web shear **buckling** coefficient.

    AISC 360-16, Eqs. G2-9, G2-10 and G2-11, Sect. G2.2, p. 16.1-72::

        h/tw <= 1.10*sqrt(kv*E/Fy):          Cv2 = 1.0                (Eq. G2-9)
        1.10*sqrt(..) < h/tw <= 1.37*sqrt(..):
            Cv2 = 1.10*sqrt(kv*E/Fy)/(h/tw)                           (Eq. G2-10)
        h/tw > 1.37*sqrt(kv*E/Fy):
            Cv2 = 1.51*kv*E/((h/tw)^2*Fy)                             (Eq. G2-11)

    Used by Sects. G3, G4 and G6 -- with different ``kv`` in each: 1.2 for angle
    legs, tee stems and weak-axis flanges, 5 for rectangular HSS and box webs.

    The elastic branch is continuous with the inelastic one at
    ``1.37*sqrt(kv*E/Fy)``, but only because ``1.51 = 1.10 * 1.37`` to three
    figures (the exact product is 1.507); the step is about 0.2%.
    """
    if h_over_tw <= 0.0:
        raise GeometryError(f"h/tw must be positive, got {h_over_tw}")
    inelastic_limit = 1.10 * math.sqrt(kv * E / Fy)
    elastic_limit = 1.37 * math.sqrt(kv * E / Fy)

    if h_over_tw <= inelastic_limit:
        return 1.0  # Eq. G2-9
    if h_over_tw <= elastic_limit:
        return inelastic_limit / h_over_tw  # Eq. G2-10
    return 1.51 * kv * E / (h_over_tw**2 * Fy)  # Eq. G2-11


def shear_area_web(d: Inch, tw: Inch) -> Inch2:
    """Web area ``Aw = d*tw``.

    AISC 360-16, Sect. G2.1, p. 16.1-70: "area of web, the overall depth times
    the web thickness".

    Note this is the **overall depth** ``d``, not the clear web depth ``h`` used
    for the slenderness ratio. Using ``h`` here understates ``Aw`` by roughly the
    two flange thicknesses -- about 6% on a W18x50.
    """
    if d <= 0.0 or tw <= 0.0:
        raise GeometryError(f"d and tw must be positive, got {d}, {tw}")
    return d * tw


# ===========================================================================
# Sect. G2.1 -- webs without tension field action
# ===========================================================================
def g2_1_strength(
    Fy: Ksi, Aw: Inch2, h_over_tw: Ratio, E: Ksi, *, kv: Ratio, rolled: bool = True
) -> tuple[Kip, Ratio, float, float]:
    """Nominal shear strength without tension field action.

    AISC 360-16, Eq. G2-1, Sect. G2.1, p. 16.1-70::

        Vn = 0.6*Fy*Aw*Cv1

    Returns ``(Vn, Cv1, phi_v, Omega_v)`` -- the factors travel with the result
    because Sect. G2.1(a) is the chapter's one exception to Sect. G1(a).
    """
    if Aw <= 0.0:
        raise GeometryError(f"Aw must be positive, got {Aw}")

    if meets_G2_1a(h_over_tw, E, Fy, rolled=rolled):
        coefficient = 1.0  # Eq. G2-2
        phi, omega = PHI_V_ROLLED, OMEGA_V_ROLLED
    else:
        coefficient = Cv1(h_over_tw, kv, E, Fy)  # Eqs. G2-3 / G2-4
        phi, omega = PHI_V, OMEGA_V

    return 0.6 * Fy * Aw * coefficient, coefficient, phi, omega


# ===========================================================================
# Sect. G2.2 -- tension field action
# ===========================================================================
def tension_field_permitted(
    a_over_h: Ratio | None,
    Aw: Inch2,
    Afc: Inch2,
    Aft: Inch2,
    h: Inch,
    bfc: Inch,
    bft: Inch,
) -> tuple[bool, bool, str]:
    """Whether Sect. G2.2 applies, and which of Eqs. G2-7 / G2-8 to use.

    AISC 360-16, Sect. G2.2, p. 16.1-72. Returns
    ``(applies, use_full_tension_field, reason)``.

    Sect. G2.2 covers **interior web panels with a/h <= 3**. Within that, the
    fuller Eq. G2-7 requires all three of::

        2*Aw/(Afc + Aft) <= 2.5
        h/bfc <= 6.0
        h/bft <= 6.0

    and Eq. G2-8 -- a weaker expression -- applies otherwise. The three tests
    exist because a tension field has to anchor against the flanges; a web too
    large relative to its flanges, or flanges too narrow relative to the web
    depth, cannot develop one fully.

    An **end panel has no adjacent panel to anchor against**, so Sect. G2.2 is
    titled "interior web panels" and end panels must be designed to Sect. G2.1.
    That is a layout fact this function cannot see -- pass ``a_over_h=None`` for
    an unstiffened web, and do not call this for an end panel.
    """
    if a_over_h is None:
        return False, False, "no transverse stiffeners: no interior panel to develop a field"
    if a_over_h > 3.0:
        return False, False, f"a/h = {a_over_h:.2f} exceeds 3.0 (Sect. G2.2 heading)"

    if min(Afc, Aft, bfc, bft, h) <= 0.0:
        raise GeometryError("flange areas, flange widths and h must all be positive")

    flange_ratio = 2.0 * Aw / (Afc + Aft)
    checks = (flange_ratio <= 2.5, h / bfc <= 6.0, h / bft <= 6.0)
    if all(checks):
        return True, True, "Eq. G2-7: 2Aw/(Afc+Aft) <= 2.5 and h/bf <= 6.0 both ways"
    failed = [
        name
        for name, ok in zip(
            (f"2Aw/(Afc+Aft) = {flange_ratio:.2f} > 2.5",
             f"h/bfc = {h / bfc:.2f} > 6.0",
             f"h/bft = {h / bft:.2f} > 6.0"),
            checks, strict=True,
        )
        if not ok
    ]
    return True, False, "Eq. G2-8: " + "; ".join(failed)


def g2_2_tension_field_strength(
    Fy: Ksi,
    Aw: Inch2,
    h_over_tw: Ratio,
    E: Ksi,
    a_over_h: Ratio,
    *,
    kv: Ratio,
    full_tension_field: bool,
) -> tuple[Kip, Ratio]:
    """Nominal shear strength considering tension field action.

    AISC 360-16, Sect. G2.2, Eqs. G2-6, G2-7 and G2-8, p. 16.1-72::

        h/tw <= 1.10*sqrt(kv*E/Fy):  Vn = 0.6*Fy*Aw                   (Eq. G2-6)

        otherwise, with the Sect. G2.2(b)(1) conditions met:
            Vn = 0.6*Fy*Aw*[Cv2 + (1 - Cv2)/(1.15*sqrt(1 + (a/h)^2))] (Eq. G2-7)

        otherwise:
            Vn = 0.6*Fy*Aw*[Cv2 + (1 - Cv2)
                            /(1.15*(a/h + sqrt(1 + (a/h)^2)))]        (Eq. G2-8)

    Returns ``(Vn, Cv2)``.

    Both expressions add a post-buckling term on top of ``Cv2``; Eq. G2-8's
    denominator is always the larger, so it always yields the smaller strength.
    They coincide only as ``a/h -> 0``.

    Sect. G2.2 closes with "The nominal shear strength is permitted to be taken
    as the larger of the values from Sections G2.1 and G2.2" -- see
    :func:`shear_strength`, which takes that maximum.
    """
    if Aw <= 0.0:
        raise GeometryError(f"Aw must be positive, got {Aw}")
    if a_over_h <= 0.0:
        raise GeometryError(f"a/h must be positive, got {a_over_h}")

    plateau = 1.10 * math.sqrt(kv * E / Fy)
    if h_over_tw <= plateau:
        return 0.6 * Fy * Aw, 1.0  # Eq. G2-6

    coefficient = Cv2(h_over_tw, kv, E, Fy)
    root = math.sqrt(1.0 + a_over_h**2)
    # Eq. G2-7 when the Sect. G2.2(b)(1) conditions hold, Eq. G2-8 otherwise;
    # Eq. G2-8's extra +a/h always makes the denominator larger and Vn smaller.
    denominator = 1.15 * root if full_tension_field else 1.15 * (a_over_h + root)
    Vn = 0.6 * Fy * Aw * (coefficient + (1.0 - coefficient) / denominator)
    return Vn, coefficient


# ===========================================================================
# Sect. G2.3 -- transverse stiffeners
# ===========================================================================
def transverse_stiffeners_required(h_over_tw: Ratio, E: Ksi, Fy: Ksi) -> bool:
    """Whether transverse stiffeners are required at all.

    AISC 360-16, Sect. G2.3(a), p. 16.1-73: not required where
    ``h/tw <= 2.46*sqrt(E/Fy)``, or where the Sect. G2.1 strength at
    ``kv = 5.34`` already exceeds the required shear.

    Only the geometric half is decided here; the strength half depends on the
    demand and is the caller's.
    """
    return h_over_tw > 2.46 * math.sqrt(E / Fy)


def stiffener_slenderness_limit(E: Ksi, Fyst: Ksi) -> Ratio:
    """Maximum ``(b/t)st`` for a transverse stiffener.

    AISC 360-16, Eq. G2-12, Sect. G2.3(d), p. 16.1-73::

        (b/t)st <= 0.56*sqrt(E/Fyst)

    ``Fyst`` is the **stiffener** yield stress, which need not match the web's.
    """
    if Fyst <= 0.0:
        raise GeometryError(f"Fyst must be positive, got {Fyst}")
    return 0.56 * math.sqrt(E / Fyst)


def stiffener_moment_of_inertia(
    h: Inch,
    tw: Inch,
    a_over_h: Ratio,
    E: Ksi,
    Fyw: Ksi,
    Fyst: Ksi,
    *,
    Vr: Kip,
    Vc1: Kip,
    Vc2: Kip,
) -> tuple[Inch4, Inch4, Inch4]:
    """Required transverse stiffener moment of inertia ``Ist``.

    AISC 360-16, Eqs. G2-13, G2-14 and G2-15, Sect. G2.3(e), p. 16.1-73.
    Returns ``(Ist_required, Ist1, Ist2)``::

        Ist  >= Ist2 + (Ist1 - Ist2)*rho_w                          (Eq. G2-13)
        Ist1 = h^4*rho_st^1.3/40 * (Fyw/E)^1.5                      (Eq. G2-14)
        Ist2 = [2.5/(a/h)^2 - 2]*bp*tw^3 >= 0.5*bp*tw^3             (Eq. G2-15)

    where ``rho_st = max(Fyw/Fyst, 1.0)``, ``bp = min(a, h)``, and
    ``rho_w = max((Vr - Vc2)/(Vc1 - Vc2), 0)``.

    ``Ist1`` develops the full post-buckling resistance; ``Ist2`` develops only
    web shear buckling. Eq. G2-13 interpolates between them on how much
    post-buckling strength the demand actually needs. The User Note permits
    ``Ist = Ist1`` conservatively.

    Note the ``2.5/(a/h)^2 - 2`` bracket goes **negative** for ``a/h > 1.118``,
    which is why Eq. G2-15 carries the ``0.5*bp*tw^3`` floor -- without it a
    widely spaced stiffener would compute a negative required inertia.
    """
    if min(h, tw) <= 0.0 or a_over_h <= 0.0:
        raise GeometryError(f"h, tw and a/h must be positive, got {h}, {tw}, {a_over_h}")
    if Vc1 <= Vc2:
        raise AISC360Error(
            f"Vc1 ({Vc1}) must exceed Vc2 ({Vc2}); Vc1 includes post-buckling strength"
        )

    rho_st = max(Fyw / Fyst, 1.0)
    Ist1 = h**4 * rho_st**1.3 / 40.0 * (Fyw / E) ** 1.5  # Eq. G2-14

    bp = min(a_over_h * h, h)
    Ist2 = max((2.5 / a_over_h**2 - 2.0) * bp * tw**3, 0.5 * bp * tw**3)  # Eq. G2-15

    rho_w = max((Vr - Vc2) / (Vc1 - Vc2), 0.0)
    Ist = Ist2 + (Ist1 - Ist2) * rho_w  # Eq. G2-13
    return Ist, Ist1, Ist2


# ===========================================================================
# Sects. G3, G4, G5, G6
# ===========================================================================
def g3_strength(Fy: Ksi, b: Inch, t: Inch, E: Ksi) -> tuple[Kip, Ratio]:
    """Nominal shear strength of a single-angle leg or a tee stem.

    AISC 360-16, Eq. G3-1, Sect. G3, p. 16.1-74::

        Vn = 0.6*Fy*b*t*Cv2

    with ``Cv2`` from Sect. G2.2 evaluated at ``h/tw = b/t`` and **kv = 1.2**.
    The low ``kv`` reflects an unstiffened element supported along one edge --
    against 5.34 for a web supported along two.
    """
    if b <= 0.0 or t <= 0.0:
        raise GeometryError(f"b and t must be positive, got {b}, {t}")
    coefficient = Cv2(b / t, 1.2, E, Fy)
    return 0.6 * Fy * b * t * coefficient, coefficient


def g4_strength(Fy: Ksi, Aw: Inch2, h_over_t: Ratio, E: Ksi) -> tuple[Kip, Ratio]:
    """Nominal shear strength of a rectangular HSS, box or other symmetric shape.

    AISC 360-16, Eq. G4-1, Sect. G4, p. 16.1-74::

        Vn = 0.6*Fy*Aw*Cv2

    with ``Cv2`` evaluated at **kv = 5** -- not 5.34. Sect. G4 names 5 exactly;
    the 5.34 of Sect. G2.1 is an unstiffened-web value that does not carry over.

    For a rectangular HSS, ``Aw = 2*h*t``: **both** webs resist the shear, and
    ``t`` is the design wall thickness of Sect. B4.2 (0.93 x nominal for A500).
    Using one web halves the strength; using the nominal wall overstates it 7.5%.
    """
    if Aw <= 0.0:
        raise GeometryError(f"Aw must be positive, got {Aw}")
    coefficient = Cv2(h_over_t, 5.0, E, Fy)
    return 0.6 * Fy * Aw * coefficient, coefficient


def g5_critical_stress(D: Inch, t: Inch, Lv: Inch, E: Ksi, Fy: Ksi) -> tuple[Ksi, str]:
    """Critical shear stress for a round HSS.

    AISC 360-16, Eqs. G5-2a and G5-2b, Sect. G5, p. 16.1-75. ``Fcr`` is the
    **larger** of::

        Fcr = 1.60*E / (sqrt(Lv/D) * (D/t)^(5/4))                   (Eq. G5-2a)
        Fcr = 0.78*E / (D/t)^(3/2)                                  (Eq. G5-2b)

    "but shall not exceed 0.6*Fy".

    Returns ``(Fcr, governing)``. Taking the larger of two buckling expressions
    looks wrong at first glance; it is not. Eq. G5-2a carries the length
    dependence and governs short tubes, Eq. G5-2b is the length-independent
    limit for long ones, and the true capacity is the higher of the two envelopes
    -- capped by shear yielding at ``0.6*Fy``.

    The User Note on p. 16.1-75 records that buckling controls only for
    ``D/t`` over 100, high-strength steels and long lengths; for standard
    sections ``Fcr = 0.6*Fy``.
    """
    if min(D, t, Lv) <= 0.0:
        raise GeometryError(f"D, t and Lv must be positive, got {D}, {t}, {Lv}")

    D_over_t = D / t
    a = 1.60 * E / (math.sqrt(Lv / D) * D_over_t**1.25)  # Eq. G5-2a
    b = 0.78 * E / D_over_t**1.5  # Eq. G5-2b
    yielding = 0.6 * Fy

    buckling = max(a, b)
    if buckling >= yielding:
        return yielding, "shear yielding (0.6*Fy governs)"
    return buckling, f"Eq. {'G5-2a' if a >= b else 'G5-2b'}"


def g5_strength(Ag: Inch2, Fcr: Ksi) -> Kip:
    """Nominal shear strength of a round HSS.

    AISC 360-16, Eq. G5-1, Sect. G5, p. 16.1-75::

        Vn = Fcr*Ag/2

    The factor of two is the shape factor for a thin circular tube in shear: the
    shear stress is not uniform round the circumference, and only about half the
    gross area is effective.
    """
    if Ag <= 0.0:
        raise GeometryError(f"Ag must be positive, got {Ag}")
    return Fcr * Ag / 2.0


def g6_strength(
    Fy: Ksi, bf: Inch, tf: Inch, E: Ksi, *, channel: bool = False
) -> tuple[Kip, Ratio]:
    """Nominal weak-axis shear strength, per shear-resisting element.

    AISC 360-16, Eq. G6-1, Sect. G6, p. 16.1-75::

        Vn = 0.6*Fy*bf*tf*Cv2

    with ``Cv2`` at **kv = 1.2** and ``h/tw`` taken as ``bf/(2*tf)`` for I-shapes
    and tees, or ``bf/tf`` for channels.

    "For each shear resisting element" -- an I-shape has **two** flanges, so the
    member strength is twice this. That multiplication is left to the caller
    because Sect. G6 states the per-element value and a channel in weak-axis
    shear may engage its flanges differently.

    The User Note records that ``Cv2 = 1.0`` for every ASTM A6 W, S, M and HP
    shape at ``Fy <= 70`` ksi.
    """
    if bf <= 0.0 or tf <= 0.0:
        raise GeometryError(f"bf and tf must be positive, got {bf}, {tf}")
    ratio = bf / tf if channel else bf / (2.0 * tf)
    coefficient = Cv2(ratio, 1.2, E, Fy)
    return 0.6 * Fy * bf * tf * coefficient, coefficient


# ===========================================================================
# Orchestrator
# ===========================================================================
def shear_strength(
    member: ShearMember,
    *,
    basis: Basis = Basis.LRFD,
    consider_tension_field: bool = False,
    Aw: Inch2 | None = None,
    Lv: Inch | None = None,
    tension_field_geometry: dict[str, float] | None = None,
) -> StrengthResult:
    """Nominal and available shear strength, ``Vn``, in kips.

    AISC 360-16, Sect. G1, p. 16.1-70: ``phi_v = 0.90`` / ``Omega_v = 1.67``
    for every provision **except** Sect. G2.1(a), which uses 1.00 / 1.50. The
    exception is carried on the limit state itself, so a report shows which
    factor produced the answer.

    Parameters
    ----------
    consider_tension_field:
        Evaluate Sect. G2.2 as well as Sect. G2.1 and report both. Sect. G2.2
        closes by permitting the **larger** of the two, so both are listed and
        the tension-field state is only added when it exceeds Sect. G2.1's.
        Applies to interior panels only -- never to an end panel.
    tension_field_geometry:
        ``Afc``, ``Aft``, ``h``, ``bfc``, ``bft`` -- needed to decide between
        Eqs. G2-7 and G2-8.
    """
    steel = member.steel
    E, Fy = steel.E, steel.Fy
    element = member.element

    if element is ShearElement.ROUND_HSS:
        props = require_properties(member.section, "Ag", "D", "t")
        if Lv is None:
            raise AISC360Error(
                "Sect. G5 needs Lv, the distance from maximum to zero shear force"
            )
        Fcr, governing = g5_critical_stress(props["D"], props["t"], Lv, E, Fy)
        state = LimitStateResult(
            LimitState.SHEAR_YIELDING if "yielding" in governing else LimitState.SHEAR_BUCKLING,
            g5_strength(props["Ag"], Fcr),
            cite("G5-1"),
            detail={"Fcr": Fcr, "D/t": props["D"] / props["t"], "Lv": Lv},
            note=governing,
        )
        return StrengthResult.build(
            "Vn", "kip", [state], phi=PHI_V, omega=OMEGA_V, basis=basis
        )

    if element is ShearElement.ANGLE_LEG_OR_TEE_STEM:
        props = require_properties(member.section, "b", "t")
        Vn, coefficient = g3_strength(Fy, props["b"], props["t"], E)
        state = LimitStateResult(
            LimitState.SHEAR_BUCKLING if coefficient < 1.0 else LimitState.SHEAR_YIELDING,
            Vn, cite("G3-1"), detail={"Cv2": coefficient, "b/t": props["b"] / props["t"]},
            note="kv = 1.2 (unstiffened element)",
        )
        return StrengthResult.build(
            "Vn", "kip", [state], phi=PHI_V, omega=OMEGA_V, basis=basis
        )

    if element is ShearElement.WEAK_AXIS_FLANGE:
        props = require_properties(member.section, "bf", "tf")
        Vn, coefficient = g6_strength(Fy, props["bf"], props["tf"], E)
        state = LimitStateResult(
            LimitState.SHEAR_BUCKLING if coefficient < 1.0 else LimitState.SHEAR_YIELDING,
            Vn, cite("G6-1"), detail={"Cv2": coefficient},
            note="per shear-resisting element; an I-shape has two flanges",
        )
        return StrengthResult.build(
            "Vn", "kip", [state], phi=PHI_V, omega=OMEGA_V, basis=basis
        )

    if element is ShearElement.RECTANGULAR_HSS:
        if Aw is None:
            raise AISC360Error("Sect. G4 needs Aw = 2*h*t for a rectangular HSS")
        Vn, coefficient = g4_strength(Fy, Aw, member.h_over_tw, E)
        state = LimitStateResult(
            LimitState.SHEAR_BUCKLING if coefficient < 1.0 else LimitState.SHEAR_YIELDING,
            Vn, cite("G4-1"), detail={"Cv2": coefficient, "Aw": Aw},
            note="kv = 5 (Sect. G4), Aw = 2*h*t with t the design wall thickness",
        )
        return StrengthResult.build(
            "Vn", "kip", [state], phi=PHI_V, omega=OMEGA_V, basis=basis
        )

    # -- Sect. G2, I-shaped members and channels ---------------------------
    if Aw is None:
        props = require_properties(member.section, "d", "tw")
        Aw = shear_area_web(props["d"], props["tw"])

    kv = kv_coefficient(member.a_over_h)
    Vn, coefficient, phi, omega = g2_1_strength(
        Fy, Aw, member.h_over_tw, E, kv=kv, rolled=member.rolled
    )
    g2_1a = meets_G2_1a(member.h_over_tw, E, Fy, rolled=member.rolled)
    states = [
        LimitStateResult(
            LimitState.SHEAR_YIELDING if coefficient == 1.0 else LimitState.SHEAR_BUCKLING,
            Vn, cite("G2-1"),
            detail={"Cv1": coefficient, "Aw": Aw, "kv": kv, "h/tw": member.h_over_tw},
            note=(
                "Sect. G2.1(a): rolled I-shape with h/tw <= 2.24*sqrt(E/Fy), phi_v = 1.00"
                if g2_1a
                else f"Sect. G2.1(b), phi_v = 0.90, kv = {kv:.2f}"
            ),
            phi=phi, omega=omega,
        )
    ]

    if consider_tension_field and member.a_over_h is not None:
        geometry = tension_field_geometry or {}
        applies, full, reason = tension_field_permitted(
            member.a_over_h, Aw,
            geometry.get("Afc", 0.0), geometry.get("Aft", 0.0),
            geometry.get("h", 0.0), geometry.get("bfc", 0.0), geometry.get("bft", 0.0),
        )
        if applies:
            Vn_tfa, cv2 = g2_2_tension_field_strength(
                Fy, Aw, member.h_over_tw, E, member.a_over_h,
                kv=kv, full_tension_field=full,
            )
            # Sect. G2.2: "permitted to be taken as the larger of the values
            # from Sections G2.1 and G2.2" -- so only add it if it wins.
            if Vn_tfa > Vn:
                states.append(
                    LimitStateResult(
                        LimitState.TENSION_FIELD_ACTION, Vn_tfa,
                        cite("G2-7" if full else "G2-8"),
                        detail={"Cv2": cv2, "a/h": member.a_over_h},
                        note=reason, phi=PHI_V, omega=OMEGA_V,
                    )
                )
                # The larger value governs, so drop the Sect. G2.1 state from
                # the comparison by reporting it as a note rather than a state.
                states = [states[-1]] + [
                    LimitStateResult(
                        states[0].limit_state, states[0].nominal, states[0].citation,
                        detail=states[0].detail,
                        note=states[0].note + "; superseded by Sect. G2.2 (larger permitted)",
                        phi=states[0].phi, omega=states[0].omega,
                    )
                ]
                return StrengthResult.build(
                    "Vn", "kip", [states[0]], phi=PHI_V, omega=OMEGA_V, basis=basis
                )

    return StrengthResult.build(
        "Vn", "kip", states, phi=PHI_V, omega=OMEGA_V, basis=basis
    )
