"""Chapter J -- Design of Connections.

Covers Sects. J1 through J10, pp. 16.1-113 to 16.1-148.

Chapter J is the widest-spread chapter in the Specification for **resistance
factors**: it uses six different phi/Omega pairs, sometimes two within a single
section. Sect. J4 alone runs tensile yielding at 0.90, tensile rupture at 0.75,
shear yielding at 1.00 and shear rupture at 0.75. Every limit state therefore
carries its own factors on the
:class:`~pyaisc360.core.result.LimitStateResult`, and
:attr:`~pyaisc360.core.result.StrengthResult.governing` compares on **available**
strength -- the machinery Chapter D forced and Chapter J needs constantly.

============================  =====  ======  ==============================
Provision                      phi   Omega   Limit state
============================  =====  ======  ==============================
J2.4 weld metal                0.75   2.00   weld rupture
J2.4 PJP weld in tension       0.80   1.88   weld rupture
J3.6 bolt tension / shear      0.75   2.00   bolt rupture
J3.8 slip, standard holes      1.00   1.50   slip
J3.8 slip, oversize            0.85   1.76   slip
J3.8 slip, long slots          0.70   2.14   slip
J3.10 bearing and tearout      0.75   2.00   bearing / tearout
J4.1 tensile yielding          0.90   1.67   yielding
J4.1 tensile rupture           0.75   2.00   rupture
J4.2 shear yielding            1.00   1.50   shear yielding
J4.2 shear rupture             0.75   2.00   shear rupture
J4.3 block shear               0.75   2.00   block shear
J7 bearing on surfaces         0.75   2.00   local compressive yielding
J8 bearing on concrete         0.65   2.31   concrete crushing
J10.1 flange local bending     0.90   1.67   flange local bending
J10.2 web local yielding       1.00   1.50   web local yielding
J10.3 web local crippling      0.75   2.00   web local crippling
J10.4 web sidesway buckling    0.85   1.76   web sidesway buckling
J10.5 web compression buckling 0.90   1.67   web compression buckling
J10.6 web panel-zone shear     0.90   1.67   panel-zone shear yielding
============================  =====  ======  ==============================

Every pair satisfies the Specification's ``Omega = 1.5/phi`` calibration, which
:func:`pyaisc360.utils.available_strength` checks.
"""

from __future__ import annotations

import math
from enum import Enum

from .core.citations import cite
from .core.config import Basis
from .core.enums import LimitState
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.result import LimitStateResult, StrengthResult
from .core.units import Inch, Inch2, Kip, Ksi, Ratio

__all__ = [
    "BoltGroup",
    "ThreadCondition",
    "HoleType",
    "FayingSurface",
    "nominal_bolt_stress",
    "minimum_pretension",
    "hole_factor",
    "slip_factors",
    # J2 welds
    "weld_length_reduction",
    "fillet_weld_effective_length",
    "directional_strength_increase",
    "fillet_weld_strength",
    "weld_group_strength",
    "base_metal_strength",
    # J3 bolts
    "bolt_strength",
    "combined_tension_shear",
    "slip_resistance",
    "slip_tension_reduction",
    "bearing_strength",
    "tearout_strength",
    "bolt_hole_strength",
    # J4 connecting elements
    "element_tension_strength",
    "element_shear_strength",
    "block_shear_strength",
    "element_compression_strength",
    # J7, J8
    "surface_bearing_strength",
    "roller_bearing_strength",
    "concrete_bearing_strength",
    # J10 concentrated forces
    "flange_local_bending",
    "web_local_yielding",
    "web_local_crippling",
    "web_sidesway_buckling",
    "web_compression_buckling",
    "panel_zone_shear",
]


# ===========================================================================
# Enumerations and tables
# ===========================================================================
class BoltGroup(str, Enum):
    """Bolt material grouping, Sect. J3.1, p. 16.1-126."""

    A307 = "ASTM A307"
    #: Group A -- F3125 Grades A325, A325M, F1852 and A354 Grade BC.
    GROUP_A = "Group A (e.g. A325)"
    #: Group B -- F3125 Grades A490, A490M, F2280 and A354 Grade BD.
    GROUP_B = "Group B (e.g. A490)"
    #: Group C -- F3043 and F3111. Limited to specific building locations and
    #: noncorrosive environments by the applicable ASTM standard.
    GROUP_C = "Group C (e.g. F3043)"
    #: Threaded parts meeting Sect. A3.4 -- stresses are fractions of Fu.
    THREADED_PART = "threaded part per Sect. A3.4"


class ThreadCondition(str, Enum):
    """Whether threads lie in a shear plane. Changes ``Fnv`` by about 25%."""

    INCLUDED = "threads not excluded from shear planes"
    EXCLUDED = "threads excluded from shear planes"


class HoleType(str, Enum):
    """Hole type, Sects. J3.8 and J3.10, Table J3.3, p. 16.1-129."""

    STANDARD = "standard"
    OVERSIZE = "oversize"
    SHORT_SLOT_PERPENDICULAR = "short-slotted, perpendicular to the load"
    SHORT_SLOT_PARALLEL = "short-slotted, parallel to the load"
    LONG_SLOT_PARALLEL = "long-slotted, parallel to the force"
    LONG_SLOT_PERPENDICULAR = "long-slotted, perpendicular to the force"


class FayingSurface(str, Enum):
    """Faying surface class, Sect. J3.8, p. 16.1-135."""

    #: Unpainted clean mill scale, or Class A coatings on blast-cleaned steel.
    CLASS_A = "Class A"
    #: Unpainted blast-cleaned steel, or Class B coatings on blast-cleaned steel.
    CLASS_B = "Class B"

    @property
    def mu(self) -> float:
        """Mean slip coefficient. Class B is 0.50 against Class A's 0.30."""
        return 0.30 if self is FayingSurface.CLASS_A else 0.50


#: Table J3.2, p. 16.1-129 -- (Fnt, Fnv) in ksi. Threaded parts carry
#: coefficients on Fu instead and are handled separately.
_TABLE_J3_2: dict[tuple[BoltGroup, ThreadCondition], tuple[Ksi, Ksi]] = {
    (BoltGroup.A307, ThreadCondition.INCLUDED): (45.0, 27.0),
    (BoltGroup.A307, ThreadCondition.EXCLUDED): (45.0, 27.0),
    (BoltGroup.GROUP_A, ThreadCondition.INCLUDED): (90.0, 54.0),
    (BoltGroup.GROUP_A, ThreadCondition.EXCLUDED): (90.0, 68.0),
    (BoltGroup.GROUP_B, ThreadCondition.INCLUDED): (113.0, 68.0),
    (BoltGroup.GROUP_B, ThreadCondition.EXCLUDED): (113.0, 84.0),
    (BoltGroup.GROUP_C, ThreadCondition.INCLUDED): (150.0, 90.0),
    (BoltGroup.GROUP_C, ThreadCondition.EXCLUDED): (150.0, 113.0),
}

#: Table J3.2 threaded-part coefficients on Fu.
_THREADED_PART_COEFFICIENTS: dict[ThreadCondition, tuple[float, float]] = {
    ThreadCondition.INCLUDED: (0.75, 0.450),
    ThreadCondition.EXCLUDED: (0.75, 0.563),
}

#: Table J3.1, p. 16.1-127 -- minimum bolt pretension Tb, kips, by nominal
#: diameter in inches. "Equal to 0.70 times the minimum tensile strength of
#: bolts ... rounded off to nearest kip."
_TABLE_J3_1: dict[float, dict[BoltGroup, float | None]] = {
    0.500: {BoltGroup.GROUP_A: 12.0, BoltGroup.GROUP_B: 15.0, BoltGroup.GROUP_C: None},
    0.625: {BoltGroup.GROUP_A: 19.0, BoltGroup.GROUP_B: 24.0, BoltGroup.GROUP_C: None},
    0.750: {BoltGroup.GROUP_A: 28.0, BoltGroup.GROUP_B: 35.0, BoltGroup.GROUP_C: None},
    0.875: {BoltGroup.GROUP_A: 39.0, BoltGroup.GROUP_B: 49.0, BoltGroup.GROUP_C: None},
    1.000: {BoltGroup.GROUP_A: 51.0, BoltGroup.GROUP_B: 64.0, BoltGroup.GROUP_C: 90.0},
    1.125: {BoltGroup.GROUP_A: 64.0, BoltGroup.GROUP_B: 80.0, BoltGroup.GROUP_C: 113.0},
    1.250: {BoltGroup.GROUP_A: 81.0, BoltGroup.GROUP_B: 102.0, BoltGroup.GROUP_C: 143.0},
    1.375: {BoltGroup.GROUP_A: 97.0, BoltGroup.GROUP_B: 121.0, BoltGroup.GROUP_C: None},
    1.500: {BoltGroup.GROUP_A: 118.0, BoltGroup.GROUP_B: 148.0, BoltGroup.GROUP_C: None},
}

#: Sect. J3.8, p. 16.1-134 -- slip resistance factors by hole type.
_SLIP_FACTORS: dict[HoleType, tuple[float, float]] = {
    HoleType.STANDARD: (1.00, 1.50),
    HoleType.SHORT_SLOT_PERPENDICULAR: (1.00, 1.50),
    HoleType.OVERSIZE: (0.85, 1.76),
    HoleType.SHORT_SLOT_PARALLEL: (0.85, 1.76),
    HoleType.LONG_SLOT_PARALLEL: (0.70, 2.14),
    HoleType.LONG_SLOT_PERPENDICULAR: (0.70, 2.14),
}

#: Sect. J2.4 and Table J2.5 -- weld metal in shear.
PHI_WELD: float = 0.75
OMEGA_WELD: float = 2.00
#: Table J2.5 -- PJP groove weld metal in tension normal to the weld axis.
PHI_WELD_PJP_TENSION: float = 0.80
OMEGA_WELD_PJP_TENSION: float = 1.88
#: Sect. J3.6 and J3.10.
PHI_BOLT: float = 0.75
OMEGA_BOLT: float = 2.00
#: Sect. J4.1(a) and J4.4.
PHI_YIELDING: float = 0.90
OMEGA_YIELDING: float = 1.67
#: Sect. J4.1(b), J4.2(b), J4.3.
PHI_RUPTURE: float = 0.75
OMEGA_RUPTURE: float = 2.00
#: Sect. J4.2(a) and J10.2 -- the phi = 1.00 cases.
PHI_SHEAR_YIELDING: float = 1.00
OMEGA_SHEAR_YIELDING: float = 1.50
#: Sect. J8 -- bearing on concrete.
PHI_CONCRETE: float = 0.65
OMEGA_CONCRETE: float = 2.31


def nominal_bolt_stress(
    group: BoltGroup, threads: ThreadCondition, *, Fu: Ksi | None = None
) -> tuple[Ksi, Ksi]:
    """Nominal tensile and shear stresses ``(Fnt, Fnv)`` from Table J3.2.

    AISC 360-16, Table J3.2, p. 16.1-129.

    Whether threads lie in a shear plane changes ``Fnv`` by about 25% -- 54 to
    68 ksi for Group A, 68 to 84 for Group B. It has **no** effect on ``Fnt``,
    because a bolt in tension is stressed through its threaded portion either
    way. ASTM A307 is the exception in the other direction: footnote [d] records
    that threads are permitted in shear planes, so both conditions give 27 ksi.

    Threaded parts meeting Sect. A3.4 take ``0.75*Fu`` and ``0.450*Fu`` or
    ``0.563*Fu``; pass ``Fu`` for those.

    Notes
    -----
    Footnote [b]: for end-loaded connections with a fastener pattern longer than
    38 in., ``Fnv`` must be reduced to 83.3% of the tabulated value. That is a
    connection-geometry fact, applied by the caller.

    Footnote [c]: A307 values reduce by 1% for each 1/16 in. over five diameters
    of grip length.
    """
    if group is BoltGroup.THREADED_PART:
        if Fu is None:
            raise AISC360Error("Table J3.2 threaded-part rows need Fu")
        if Fu <= 0.0:
            raise GeometryError(f"Fu must be positive, got {Fu}")
        tensile, shear = _THREADED_PART_COEFFICIENTS[threads]
        return tensile * Fu, shear * Fu
    return _TABLE_J3_2[(group, threads)]


def minimum_pretension(group: BoltGroup, diameter: Inch) -> Kip:
    """Minimum bolt pretension ``Tb`` from Table J3.1.

    AISC 360-16, Table J3.1, p. 16.1-127. Equal to 0.70 times the bolt's minimum
    tensile strength, rounded to the nearest kip.

    Required for pretensioned and slip-critical connections, and it is ``Tb``
    -- not the applied bolt force -- that enters Eq. J3-4.

    Raises
    ------
    AISC360Error
        For a diameter or group Table J3.1 does not list. Group C is tabulated
        only for 1, 1-1/8 and 1-1/4 in.
    """
    if diameter not in _TABLE_J3_1:
        listed = ", ".join(f"{d:g}" for d in sorted(_TABLE_J3_1))
        raise AISC360Error(
            f"Table J3.1 does not list a {diameter:g} in. bolt. Listed: {listed}"
        )
    value = _TABLE_J3_1[diameter].get(group)
    if value is None:
        raise AISC360Error(
            f"Table J3.1 gives no pretension for {group.value} at {diameter:g} in."
        )
    return value


def hole_factor(hole: HoleType) -> float:
    """Hole factor ``hf`` for Eq. J3-4.

    AISC 360-16, Sect. J3.8, p. 16.1-135: ``hf = 1.0`` where fillers are not
    used, reducing where they are. Returned as 1.0 here; filler cases are a
    connection-detail input.
    """
    return 1.0


def slip_factors(hole: HoleType) -> tuple[float, float]:
    """``(phi, Omega)`` for slip resistance by hole type.

    AISC 360-16, Sect. J3.8, p. 16.1-134::

        standard and short slots perpendicular:  phi = 1.00, Omega = 1.50
        oversize and short slots parallel:       phi = 0.85, Omega = 1.76
        long slots:                              phi = 0.70, Omega = 2.14

    Slip is a **serviceability-like** limit state for a connection that is still
    safe against rupture afterwards, which is why standard holes carry
    ``phi = 1.00`` -- the same reasoning as Sect. G2.1(a).
    """
    return _SLIP_FACTORS[hole]


# ===========================================================================
# Sect. J2 -- Welds
# ===========================================================================
def weld_length_reduction(length: Inch, weld_size: Inch) -> Ratio:
    """Reduction factor ``beta`` for a long end-loaded fillet weld.

    AISC 360-16, Eq. J2-1, Sect. J2.2b(d), p. 16.1-120::

        beta = 1.2 - 0.002*(l/w)  <= 1.0

    Applies when ``l > 100*w``. Shear lag along a long weld means the far end
    never reaches its strength before the near end tears.
    """
    if weld_size <= 0.0 or length <= 0.0:
        raise GeometryError(f"length and weld size must be positive, got {length}, {weld_size}")
    return min(1.2 - 0.002 * (length / weld_size), 1.0)


def fillet_weld_effective_length(length: Inch, weld_size: Inch) -> tuple[Inch, str]:
    """Effective length of an end-loaded fillet weld.

    AISC 360-16, Sect. J2.2b(d), p. 16.1-120, in three ranges::

        l <= 100*w:  effective length = l
        l <= 300*w:  effective length = beta*l,  beta from Eq. J2-1
        l >  300*w:  effective length = 180*w

    The third branch is a **hard cap, not a continuation**: at ``l = 300*w``,
    Eq. J2-1 gives ``beta = 0.6`` and ``beta*l = 180*w`` -- so the two meet
    exactly, and beyond that point adding weld adds nothing at all.
    """
    if weld_size <= 0.0 or length <= 0.0:
        raise GeometryError(f"length and weld size must be positive, got {length}, {weld_size}")

    ratio = length / weld_size
    if ratio <= 100.0:
        return length, "l <= 100w: full length is effective"
    if ratio <= 300.0:
        beta = weld_length_reduction(length, weld_size)
        return beta * length, f"Eq. J2-1: beta = {beta:.3f}"
    return 180.0 * weld_size, "l > 300w: capped at 180w -- more weld adds nothing"


def directional_strength_increase(theta_degrees: float) -> float:
    """Directional strength increase for a fillet weld.

    AISC 360-16, Sect. J2.4(b) and Eq. J2-5, p. 16.1-122::

        1.0 + 0.50*sin^1.5(theta)

    ``theta`` is the angle between the line of action of the force and the weld
    **longitudinal axis**. A longitudinal weld (0 degrees) gets nothing; a
    transverse weld (90 degrees) gets the full 1.5x, because a transversely
    loaded fillet fails on a plane rotated away from the 45-degree throat and
    mobilises more metal.

    Permitted only "if strain compatibility of the various weld elements is
    considered" -- so it may not simply be applied to a mixed group, which is
    what Eq. J2-6b exists to handle.
    """
    if not 0.0 <= theta_degrees <= 90.0:
        raise GeometryError(f"theta must lie in [0, 90] degrees, got {theta_degrees}")
    # math.pow, not **: mypy widens ** to Any because a negative base with a
    # fractional exponent is complex. sin(theta) >= 0 on [0, 90] regardless.
    return 1.0 + 0.50 * math.pow(math.sin(math.radians(theta_degrees)), 1.5)


def fillet_weld_strength(
    FEXX: Ksi,
    Awe: Inch2,
    *,
    theta_degrees: float = 0.0,
    directional_increase: bool = False,
    basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Weld metal rupture strength of a fillet weld.

    AISC 360-16, Eqs. J2-3, J2-4 and J2-5, Sect. J2.4, p. 16.1-122, with
    Table J2.5, p. 16.1-123::

        Rn  = Fnw*Awe                                             (J2-3, J2-4)
        Fnw = 0.60*FEXX                                     (Table J2.5)
        Fnw = 0.60*FEXX*(1.0 + 0.50*sin^1.5(theta))               (J2-5)

    with ``phi = 0.75`` / ``Omega = 2.00``.

    ``Awe`` is the effective throat area -- for a fillet, ``0.707*w`` times the
    effective length, not the leg times the length. Using the leg overstates the
    strength by 41%.

    Parameters
    ----------
    directional_increase:
        Apply Eq. J2-5. Sect. J2.4(b) permits it only when strain compatibility
        is considered, so it is off by default.
    """
    if FEXX <= 0.0 or Awe <= 0.0:
        raise GeometryError(f"FEXX and Awe must be positive, got {FEXX}, {Awe}")

    if directional_increase:
        factor = directional_strength_increase(theta_degrees)
        Fnw = 0.60 * FEXX * factor  # Eq. J2-5
        citation, note = cite("J2-5"), f"directional increase {factor:.3f} at {theta_degrees:g} deg"
    else:
        Fnw = 0.60 * FEXX  # Table J2.5
        citation, note = cite("J2-4"), "Table J2.5: Fnw = 0.60*FEXX, no directional increase"

    state = LimitStateResult(
        LimitState.WELD_RUPTURE, Fnw * Awe, citation,
        detail={"Fnw": Fnw, "Awe": Awe, "FEXX": FEXX}, note=note,
        phi=PHI_WELD, omega=OMEGA_WELD,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_WELD, omega=OMEGA_WELD, basis=basis
    )


def weld_group_strength(Rnwl: Kip, Rnwt: Kip) -> tuple[Kip, str]:
    """Combined strength of a concentrically loaded fillet weld group.

    AISC 360-16, Eqs. J2-6a and J2-6b, Sect. J2.4(b)(2), p. 16.1-124. The
    **greater** of::

        Rn = Rnwl + Rnwt                                          (J2-6a)
        Rn = 0.85*Rnwl + 1.5*Rnwt                                 (J2-6b)

    For weld groups with uniform leg size loaded both longitudinally and
    transversely.

    Parameters
    ----------
    Rnwt:
        Total nominal strength of the **transversely** loaded welds, computed
        from Table J2.5 **without** the Sect. J2.4(b) directional increase.
        Eq. J2-6b's 1.5 factor *is* that increase, applied at the group level
        with an 0.85 penalty on the longitudinal welds to account for the strain
        incompatibility between the two orientations. Applying both would
        double-count.
    """
    if Rnwl < 0.0 or Rnwt < 0.0:
        raise GeometryError(f"weld strengths must be non-negative, got {Rnwl}, {Rnwt}")

    simple = Rnwl + Rnwt  # Eq. J2-6a
    redistributed = 0.85 * Rnwl + 1.5 * Rnwt  # Eq. J2-6b
    if redistributed >= simple:
        return redistributed, "Eq. J2-6b governs (transverse welds dominate)"
    return simple, "Eq. J2-6a governs (longitudinal welds dominate)"


def base_metal_strength(
    FnBM: Ksi, ABM: Inch2, *, basis: Basis = Basis.LRFD, phi: float = PHI_RUPTURE,
    omega: float = OMEGA_RUPTURE,
) -> StrengthResult:
    """Base metal strength at a weld.

    AISC 360-16, Eq. J2-2, Sect. J2.4(a), p. 16.1-122::

        Rn = FnBM*ABM

    Sect. J2.4(a) requires the **lower** of this and the weld metal strength.
    ``FnBM`` and its factors come from Table J2.5 and Sect. J4 -- for shear
    rupture of the base metal, ``FnBM = 0.60*Fu`` at ``phi = 0.75``; for shear
    yielding, ``0.60*Fy`` at ``phi = 1.00``.
    """
    if FnBM <= 0.0 or ABM <= 0.0:
        raise GeometryError(f"FnBM and ABM must be positive, got {FnBM}, {ABM}")
    state = LimitStateResult(
        LimitState.BASE_METAL_RUPTURE, FnBM * ABM, cite("J2-2"),
        detail={"FnBM": FnBM, "ABM": ABM},
        note="Sect. J2.4(a): take the lower of base metal and weld metal strength",
        phi=phi, omega=omega,
    )
    return StrengthResult.build("Rn", "kip", [state], phi=phi, omega=omega, basis=basis)


# ===========================================================================
# Sect. J3 -- Bolts and Threaded Parts
# ===========================================================================
def bolt_strength(
    Fn: Ksi, Ab: Inch2, *, limit_state: LimitState = LimitState.BOLT_SHEAR,
    basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Tensile or shear rupture strength of one bolt.

    AISC 360-16, Eq. J3-1, Sect. J3.6, p. 16.1-131::

        Rn = Fn*Ab

    with ``phi = 0.75`` / ``Omega = 2.00``.

    ``Ab`` is the **nominal unthreaded body area** -- ``pi*d^2/4`` on the full
    diameter, not the tensile stress area. The thread reduction is already
    inside ``Fn``, which is why Table J3.2 distinguishes threads included from
    excluded. Using the stress area here double-counts it.

    Sect. J3.6 adds that "the required tensile strength shall include any tension
    resulting from prying action produced by deformation of the connected parts"
    -- prying is not in this equation and must be added to the demand.
    """
    if Fn <= 0.0 or Ab <= 0.0:
        raise GeometryError(f"Fn and Ab must be positive, got {Fn}, {Ab}")
    state = LimitStateResult(
        limit_state, Fn * Ab, cite("J3-1"),
        detail={"Fn": Fn, "Ab": Ab},
        note="Ab is the nominal UNTHREADED body area; prying must be added to the demand",
        phi=PHI_BOLT, omega=OMEGA_BOLT,
    )
    return StrengthResult.build("Rn", "kip", [state], phi=PHI_BOLT, omega=OMEGA_BOLT, basis=basis)


def combined_tension_shear(
    Fnt: Ksi, Fnv: Ksi, frv: Ksi, basis: Basis = Basis.LRFD
) -> tuple[Ksi, str]:
    """Nominal tensile stress modified for concurrent shear.

    AISC 360-16, Eqs. J3-3a and J3-3b, Sect. J3.7, p. 16.1-134::

        LRFD:  Fnt' = 1.3*Fnt - (Fnt/(phi*Fnv))*frv  <= Fnt       (J3-3a)
        ASD:   Fnt' = 1.3*Fnt - (Omega*Fnt/Fnv)*frv  <= Fnt       (J3-3b)

    with ``phi = 0.75`` / ``Omega = 2.00``. The modified stress then goes into
    Eq. J3-2, ``Rn = Fnt'*Ab``.

    Note the **1.3 multiplier**: the elliptical interaction of tension and shear
    is approximated by a straight line that starts *above* ``Fnt``, then capped
    back at ``Fnt``. So a small shear costs nothing at all -- which is what the
    User Note means by "when the required stress ... is less than or equal to
    30% of the corresponding available stress, the effects of combined stress
    need not be investigated".

    The available shear stress must still equal or exceed ``frv`` independently;
    this equation only reduces the tensile capacity.
    """
    if Fnt <= 0.0 or Fnv <= 0.0:
        raise GeometryError(f"Fnt and Fnv must be positive, got {Fnt}, {Fnv}")
    if frv < 0.0:
        raise GeometryError(f"required shear stress must be non-negative, got {frv}")

    if basis.is_lrfd:
        modified = 1.3 * Fnt - (Fnt / (PHI_BOLT * Fnv)) * frv  # Eq. J3-3a
        equation = "J3-3a"
    else:
        modified = 1.3 * Fnt - (OMEGA_BOLT * Fnt / Fnv) * frv  # Eq. J3-3b
        equation = "J3-3b"

    capped = min(modified, Fnt)
    note = f"Eq. {equation}"
    if capped >= Fnt:
        note += "; capped at Fnt -- the shear is small enough to cost nothing"
    return max(capped, 0.0), note


def slip_resistance(
    mu: float, Du: float, hf: float, Tb: Kip, ns: int, hole: HoleType,
    *, basis: Basis = Basis.LRFD, ksc: float = 1.0,
) -> StrengthResult:
    """Slip resistance of one bolt in a slip-critical connection.

    AISC 360-16, Eq. J3-4, Sect. J3.8, p. 16.1-134::

        Rn = mu*Du*hf*Tb*ns

    Parameters
    ----------
    mu:
        Mean slip coefficient -- 0.30 for Class A faying surfaces, 0.50 for
        Class B. See :attr:`FayingSurface.mu`.
    Du:
        Ratio of mean installed pretension to the specified minimum. Sect. J3.8
        gives 1.13 unless another value is approved.
    hf:
        Factor for fillers.
    Tb:
        **Minimum** pretension from Table J3.1 -- not the applied bolt force.
    ns:
        Number of slip planes.
    ksc:
        Reduction from Eq. J3-5a/J3-5b for concurrent applied tension; see
        :func:`slip_tension_reduction`.

    The factors come from the hole type, not from the chapter: standard holes
    take ``phi = 1.00``, oversize 0.85, long slots 0.70.
    """
    for name, value in (("mu", mu), ("Du", Du), ("hf", hf), ("Tb", Tb)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    if ns < 1:
        raise GeometryError(f"there must be at least one slip plane, got {ns}")
    if not 0.0 <= ksc <= 1.0:
        raise GeometryError(f"ksc must lie in [0, 1], got {ksc}")

    phi, omega = slip_factors(hole)
    state = LimitStateResult(
        LimitState.SLIP, mu * Du * hf * Tb * ns * ksc, cite("J3-4"),
        detail={"mu": mu, "Du": Du, "hf": hf, "Tb": Tb, "ns": float(ns), "ksc": ksc},
        note=f"{hole.value} holes: phi = {phi:.2f}, Omega = {omega:.2f}",
        phi=phi, omega=omega,
    )
    return StrengthResult.build("Rn", "kip", [state], phi=phi, omega=omega, basis=basis)


def slip_tension_reduction(T: Kip, Du: float, Tb: Kip, nb: int, basis: Basis) -> float:
    """Slip resistance reduction ``ksc`` for concurrent applied tension.

    AISC 360-16, Eqs. J3-5a and J3-5b, Sect. J3.9, p. 16.1-135::

        LRFD:  ksc = 1 - Tu/(Du*Tb*nb)      >= 0                  (J3-5a)
        ASD:   ksc = 1 - 1.5*Ta/(Du*Tb*nb)  >= 0                  (J3-5b)

    Applied tension relieves the clamping force that generates friction, so slip
    resistance falls. The ASD form carries **1.5**, not the 1.6 used elsewhere
    for ASD force-level adjustment -- Sect. J3.9 states 1.5 explicitly.

    Floored at zero: once the applied tension reaches the total pretension there
    is no clamping left, and a negative slip resistance is meaningless.
    """
    if Du <= 0.0 or Tb <= 0.0:
        raise GeometryError(f"Du and Tb must be positive, got {Du}, {Tb}")
    if nb < 1:
        raise GeometryError(f"there must be at least one bolt, got {nb}")

    factor = 1.0 if basis.is_lrfd else 1.5
    return max(1.0 - factor * abs(T) / (Du * Tb * nb), 0.0)


def bearing_strength(
    d: Inch, t: Inch, Fu: Ksi, *, deformation_considered: bool = True,
    long_slot_perpendicular: bool = False,
) -> tuple[Kip, str]:
    """Bearing strength at a bolt hole.

    AISC 360-16, Eqs. J3-6a, J3-6b and J3-6e, Sect. J3.10, p. 16.1-136::

        deformation is a design consideration:      Rn = 2.4*d*t*Fu   (J3-6a)
        deformation is not:                         Rn = 3.0*d*t*Fu   (J3-6b)
        long slot perpendicular to the force:       Rn = 2.0*d*t*Fu   (J3-6e)

    "Deformation at the bolt hole at service load is a design consideration" is
    the normal case and the default -- the 3.0 branch permits a hole to ovalise
    by more than 1/4 in., which most connections cannot tolerate.
    """
    if min(d, t, Fu) <= 0.0:
        raise GeometryError(f"d, t and Fu must be positive, got {d}, {t}, {Fu}")

    if long_slot_perpendicular:
        return 2.0 * d * t * Fu, "Eq. J3-6e, long slot perpendicular to the force"
    if deformation_considered:
        return 2.4 * d * t * Fu, "Eq. J3-6a, deformation is a design consideration"
    return 3.0 * d * t * Fu, "Eq. J3-6b, deformation is not a design consideration"


def tearout_strength(
    lc: Inch, t: Inch, Fu: Ksi, *, deformation_considered: bool = True,
    long_slot_perpendicular: bool = False,
) -> tuple[Kip, str]:
    """Tearout strength at a bolt hole.

    AISC 360-16, Eqs. J3-6c, J3-6d and J3-6f, Sect. J3.10, p. 16.1-136::

        deformation is a design consideration:      Rn = 1.2*lc*t*Fu  (J3-6c)
        deformation is not:                         Rn = 1.5*lc*t*Fu  (J3-6d)
        long slot perpendicular to the force:       Rn = 1.0*lc*t*Fu  (J3-6f)

    ``lc`` is the **clear distance**, in the direction of the force, between the
    edge of the hole and the edge of the adjacent hole or of the material -- not
    the centre-to-centre spacing or the edge distance. Using the centre distance
    overstates tearout by roughly half a hole diameter's worth.
    """
    if min(lc, t, Fu) <= 0.0:
        raise GeometryError(f"lc, t and Fu must be positive, got {lc}, {t}, {Fu}")

    if long_slot_perpendicular:
        return 1.0 * lc * t * Fu, "Eq. J3-6f, long slot perpendicular to the force"
    if deformation_considered:
        return 1.2 * lc * t * Fu, "Eq. J3-6c, deformation is a design consideration"
    return 1.5 * lc * t * Fu, "Eq. J3-6d, deformation is not a design consideration"


def bolt_hole_strength(
    d: Inch, lc: Inch, t: Inch, Fu: Ksi, *, deformation_considered: bool = True,
    long_slot_perpendicular: bool = False, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Combined bearing and tearout at one bolt hole, Sect. J3.10, p. 16.1-135.

    Returns both limit states; the lower governs. The User Note to Sect. J3.6
    records the way these combine at the group level: "The effective strength of
    an individual fastener may be taken as the lesser of the fastener shear
    strength per Section J3.6 or the bearing or tearout strength at the bolt hole
    per Section J3.10. The strength of the bolt group is taken as the sum of the
    effective strengths of the individual fasteners."

    That per-bolt minimum is the important part: an edge bolt limited by tearout
    and an interior bolt limited by shear contribute *different* amounts, so a
    group cannot be checked by multiplying one bolt's strength by the count.
    """
    bearing, bearing_note = bearing_strength(
        d, t, Fu, deformation_considered=deformation_considered,
        long_slot_perpendicular=long_slot_perpendicular,
    )
    tearout, tearout_note = tearout_strength(
        lc, t, Fu, deformation_considered=deformation_considered,
        long_slot_perpendicular=long_slot_perpendicular,
    )
    states = [
        LimitStateResult(
            LimitState.BOLT_BEARING, bearing, cite("J3-6a"),
            detail={"d": d, "t": t, "Fu": Fu}, note=bearing_note,
            phi=PHI_BOLT, omega=OMEGA_BOLT,
        ),
        LimitStateResult(
            LimitState.BOLT_TEAROUT, tearout, cite("J3-6c"),
            detail={"lc": lc, "t": t, "Fu": Fu},
            note=tearout_note + "; lc is the CLEAR distance, not centre-to-centre",
            phi=PHI_BOLT, omega=OMEGA_BOLT,
        ),
    ]
    return StrengthResult.build("Rn", "kip", states, phi=PHI_BOLT, omega=OMEGA_BOLT, basis=basis)


# ===========================================================================
# Sect. J4 -- Affected Elements and Connecting Elements
# ===========================================================================
def element_tension_strength(
    Fy: Ksi, Ag: Inch2, Fu: Ksi, Ae: Inch2, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Tensile strength of a connecting element.

    AISC 360-16, Eqs. J4-1 and J4-2, Sect. J4.1, p. 16.1-137::

        tensile yielding:  Rn = Fy*Ag,  phi = 0.90, Omega = 1.67   (J4-1)
        tensile rupture:   Rn = Fu*Ae,  phi = 0.75, Omega = 2.00   (J4-2)

    The same two limit states and the same split factors as Sect. D2, so the
    governing state must again be chosen on **available** strength.

    The User Note records that "the effective net area of the connection plate
    may be limited due to stress distribution as calculated by methods such as
    the Whitmore section" -- ``Ae`` is not simply the net area of the plate.
    """
    states = [
        LimitStateResult(
            LimitState.TENSILE_YIELDING, Fy * Ag, cite("J4-1"),
            detail={"Fy": Fy, "Ag": Ag}, note="yielding of the gross section",
            phi=PHI_YIELDING, omega=OMEGA_YIELDING,
        ),
        LimitStateResult(
            LimitState.TENSILE_RUPTURE, Fu * Ae, cite("J4-2"),
            detail={"Fu": Fu, "Ae": Ae},
            note="Ae may be limited by the Whitmore section, not just by the holes",
            phi=PHI_RUPTURE, omega=OMEGA_RUPTURE,
        ),
    ]
    return StrengthResult.build(
        "Rn", "kip", states, phi=PHI_YIELDING, omega=OMEGA_YIELDING, basis=basis
    )


def element_shear_strength(
    Fy: Ksi, Agv: Inch2, Fu: Ksi, Anv: Inch2, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Shear strength of a connecting element.

    AISC 360-16, Eqs. J4-3 and J4-4, Sect. J4.2, p. 16.1-137::

        shear yielding:  Rn = 0.60*Fy*Agv,  phi = 1.00, Omega = 1.50  (J4-3)
        shear rupture:   Rn = 0.60*Fu*Anv,  phi = 0.75, Omega = 2.00  (J4-4)

    Shear yielding carries ``phi = 1.00`` -- one of only three places in the
    Specification, with Sect. G2.1(a) and Sect. J10.2. The factors differ so
    sharply between the two states here (1.00 against 0.75) that comparing
    nominal strengths gives the wrong answer over a wide range of ``Anv/Agv``.
    """
    states = [
        LimitStateResult(
            LimitState.SHEAR_YIELDING, 0.60 * Fy * Agv, cite("J4-3"),
            detail={"Fy": Fy, "Agv": Agv}, note="phi = 1.00 for shear yielding",
            phi=PHI_SHEAR_YIELDING, omega=OMEGA_SHEAR_YIELDING,
        ),
        LimitStateResult(
            LimitState.SHEAR_RUPTURE, 0.60 * Fu * Anv, cite("J4-4"),
            detail={"Fu": Fu, "Anv": Anv}, note="phi = 0.75 for shear rupture",
            phi=PHI_RUPTURE, omega=OMEGA_RUPTURE,
        ),
    ]
    return StrengthResult.build(
        "Rn", "kip", states, phi=PHI_SHEAR_YIELDING, omega=OMEGA_SHEAR_YIELDING, basis=basis
    )


def block_shear_strength(
    Fu: Ksi, Anv: Inch2, Ant: Inch2, Fy: Ksi, Agv: Inch2, Ubs: float = 1.0,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Block shear rupture strength.

    AISC 360-16, Eq. J4-5, Sect. J4.3, p. 16.1-138::

        Rn = 0.60*Fu*Anv + Ubs*Fu*Ant  <=  0.60*Fy*Agv + Ubs*Fu*Ant

    with ``phi = 0.75`` / ``Omega = 2.00``.

    The structure is easy to mis-read: the **tension term is identical on both
    sides**. Only the shear term is capped -- shear rupture on the net area is
    limited by shear yielding on the gross area. It is not a comparison of two
    independent expressions.

    ``Ubs = 1.0`` where the tension stress is uniform, ``0.5`` where it is not.
    The Commentary illustrates the nonuniform cases; a coped beam with a single
    row of bolts is the usual one.
    """
    for name, value in (("Anv", Anv), ("Ant", Ant), ("Agv", Agv)):
        if value < 0.0:
            raise GeometryError(f"{name} must be non-negative, got {value}")
    if Ubs not in (0.5, 1.0):
        raise AISC360Error(f"Ubs is 1.0 (uniform tension) or 0.5 (nonuniform), got {Ubs}")

    tension_term = Ubs * Fu * Ant
    shear_rupture = 0.60 * Fu * Anv
    shear_yielding = 0.60 * Fy * Agv
    governing_shear = min(shear_rupture, shear_yielding)

    state = LimitStateResult(
        LimitState.BLOCK_SHEAR, governing_shear + tension_term, cite("J4-5"),
        detail={
            "0.60*Fu*Anv": shear_rupture,
            "0.60*Fy*Agv": shear_yielding,
            "Ubs*Fu*Ant": tension_term,
            "Ubs": Ubs,
        },
        note=(
            "shear rupture governs" if shear_rupture <= shear_yielding
            else "shear yielding caps the shear term"
        ),
        phi=PHI_RUPTURE, omega=OMEGA_RUPTURE,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_RUPTURE, omega=OMEGA_RUPTURE, basis=basis
    )


def element_compression_strength(
    Fy: Ksi, Ag: Inch2, Lc_over_r: Ratio, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Compressive strength of a connecting element.

    AISC 360-16, Eq. J4-6, Sect. J4.4, p. 16.1-138::

        Lc/r <= 25:  Pn = Fy*Ag,  phi = 0.90, Omega = 1.67         (J4-6)
        Lc/r >  25:  the provisions of Chapter E apply

    The ``Lc/r <= 25`` shortcut exists because a stub that short cannot buckle
    before it squashes -- Eq. E3-2 at ``Lc/r = 25`` already gives
    ``Fcr = 0.97*Fy``, so the 3% error is not worth a buckling calculation.

    Raises
    ------
    OutOfScopeError
        Above 25, where Sect. J4.4(b) directs the caller to Chapter E rather
        than extending Eq. J4-6.
    """
    if Lc_over_r < 0.0:
        raise GeometryError(f"Lc/r must be non-negative, got {Lc_over_r}")
    if Lc_over_r > 25.0:
        raise OutOfScopeError(
            f"Lc/r = {Lc_over_r:.1f} exceeds 25. Sect. J4.4(b), p. 16.1-138, directs "
            "that the provisions of Chapter E apply -- use "
            "pyaisc360.chapter_e.compressive_strength."
        )
    state = LimitStateResult(
        LimitState.YIELDING, Fy * Ag, cite("J4-6"),
        detail={"Fy": Fy, "Ag": Ag, "Lc/r": Lc_over_r},
        note="Lc/r <= 25: too stocky to buckle before squashing",
        phi=PHI_YIELDING, omega=OMEGA_YIELDING,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_YIELDING, omega=OMEGA_YIELDING, basis=basis
    )


# ===========================================================================
# Sects. J7 and J8 -- Bearing
# ===========================================================================
def surface_bearing_strength(Fy: Ksi, Apb: Inch2, *, basis: Basis = Basis.LRFD) -> StrengthResult:
    """Bearing strength of milled surfaces, pins and fitted stiffeners.

    AISC 360-16, Eq. J7-1, Sect. J7(a), p. 16.1-140::

        Rn = 1.8*Fy*Apb

    with ``phi = 0.75`` / ``Omega = 2.00``. The 1.8 factor reflects the confined
    yielding of a surface bearing on another -- material cannot spread sideways,
    so it carries well above ``Fy``.
    """
    if Fy <= 0.0 or Apb <= 0.0:
        raise GeometryError(f"Fy and Apb must be positive, got {Fy}, {Apb}")
    state = LimitStateResult(
        LimitState.BOLT_BEARING, 1.8 * Fy * Apb, cite("J7-1"),
        detail={"Fy": Fy, "Apb": Apb}, note="finished surfaces, pins in reamed holes, "
        "fitted bearing stiffeners",
        phi=PHI_BOLT, omega=OMEGA_BOLT,
    )
    return StrengthResult.build("Rn", "kip", [state], phi=PHI_BOLT, omega=OMEGA_BOLT, basis=basis)


def roller_bearing_strength(Fy: Ksi, lb: Inch, d: Inch) -> tuple[Kip, str]:
    """Bearing strength of an expansion roller or rocker.

    AISC 360-16, Eqs. J7-2 and J7-3, Sect. J7(b), p. 16.1-140::

        d <= 25 in.:  Rn = 1.2*(Fy - 13)*lb*d/20                   (J7-2)
        d >  25 in.:  Rn = 6.0*(Fy - 13)*lb*sqrt(d)/20             (J7-3)

    The ``Fy - 13`` offset is dimensional -- these are empirical expressions in
    ksi and inches, not general formulae, and the metric forms (Eqs. J7-2M and
    J7-3M) use ``Fy - 90`` with different coefficients rather than a unit
    conversion of these.

    Note the change of form at 25 in.: the small-roller expression is linear in
    ``d`` and the large-roller one goes as ``sqrt(d)``.
    """
    if Fy <= 13.0:
        raise OutOfScopeError(
            f"Eqs. J7-2/J7-3 subtract 13 ksi from Fy and are meaningless at "
            f"Fy = {Fy:g} ksi. They are empirical expressions in ksi and inches."
        )
    if lb <= 0.0 or d <= 0.0:
        raise GeometryError(f"lb and d must be positive, got {lb}, {d}")

    if d <= 25.0:
        return 1.2 * (Fy - 13.0) * lb * d / 20.0, "Eq. J7-2, d <= 25 in."
    return 6.0 * (Fy - 13.0) * lb * math.sqrt(d) / 20.0, "Eq. J7-3, d > 25 in."


def concrete_bearing_strength(
    fc_prime: Ksi, A1: Inch2, A2: Inch2 | None = None, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Bearing strength on concrete beneath a column base plate.

    AISC 360-16, Eqs. J8-1 and J8-2, Sect. J8, p. 16.1-141::

        full area of support:  Pp = 0.85*fc'*A1                    (J8-1)
        less than full area:   Pp = 0.85*fc'*A1*sqrt(A2/A1)
                                   <= 1.7*fc'*A1                  (J8-2)

    with ``phi_c = 0.65`` / ``Omega_c = 2.31`` -- the only place in the
    Specification with those values, because the limit state is crushing of
    *concrete*, not steel.

    ``A2`` is "the maximum area of the portion of the supporting surface that is
    geometrically **similar to and concentric with** the loaded area" -- not the
    whole footing. The ``sqrt(A2/A1)`` confinement bonus is capped at 2, which
    is what the ``1.7*fc'*A1`` limit expresses.

    Sect. J8 applies "in the absence of code regulations"; ACI 318 governs where
    it applies.
    """
    if fc_prime <= 0.0 or A1 <= 0.0:
        raise GeometryError(f"fc' and A1 must be positive, got {fc_prime}, {A1}")

    if A2 is None or A2 <= A1:
        Pp = 0.85 * fc_prime * A1  # Eq. J8-1
        note = "Eq. J8-1: bearing on the full area of the support"
    else:
        confinement = math.sqrt(A2 / A1)
        Pp = min(0.85 * fc_prime * A1 * confinement, 1.7 * fc_prime * A1)  # Eq. J8-2
        capped = confinement > 2.0
        note = (
            f"Eq. J8-2: sqrt(A2/A1) = {confinement:.3f}"
            + ("; capped at 1.7*fc'*A1 (confinement bonus limited to 2)" if capped else "")
        )

    state = LimitStateResult(
        LimitState.CONCRETE_BEARING, Pp, cite("J8-1" if A2 is None or A2 <= A1 else "J8-2"),
        detail={"fc'": fc_prime, "A1": A1, **({"A2": A2} if A2 else {})}, note=note,
        phi=PHI_CONCRETE, omega=OMEGA_CONCRETE,
    )
    return StrengthResult.build(
        "Pp", "kip", [state], phi=PHI_CONCRETE, omega=OMEGA_CONCRETE, basis=basis
    )


# ===========================================================================
# Sect. J10 -- Flanges and Webs with Concentrated Forces
# ===========================================================================
def flange_local_bending(
    Fyf: Ksi, tf: Inch, *, near_member_end: bool = False, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Flange local bending under a tensile concentrated force.

    AISC 360-16, Eq. J10-1, Sect. J10.1, p. 16.1-142::

        Rn = 6.25*Fyf*tf^2

    with ``phi = 0.90`` / ``Omega = 1.67``.

    Depends on the flange thickness **squared** and on nothing else -- not on
    the flange width, not on the bearing length. Two reliefs apply:

    * if the length of loading across the flange is less than ``0.15*bf``,
      Eq. J10-1 need not be checked at all;
    * if the force is applied less than ``10*tf`` from the member end, ``Rn``
      is **reduced by 50%**.
    """
    if Fyf <= 0.0 or tf <= 0.0:
        raise GeometryError(f"Fyf and tf must be positive, got {Fyf}, {tf}")

    Rn = 6.25 * Fyf * tf**2  # Eq. J10-1
    note = "Eq. J10-1 need not be checked if the loaded length is under 0.15*bf"
    if near_member_end:
        Rn *= 0.5
        note = "reduced 50%: force applied within 10*tf of the member end; " + note

    state = LimitStateResult(
        LimitState.FLANGE_LOCAL_BENDING, Rn, cite("J10-1"),
        detail={"Fyf": Fyf, "tf": tf}, note=note,
        phi=PHI_YIELDING, omega=OMEGA_YIELDING,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_YIELDING, omega=OMEGA_YIELDING, basis=basis
    )


def web_local_yielding(
    Fyw: Ksi, tw: Inch, k: Inch, lb: Inch, *, near_member_end: bool = False,
    basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Web local yielding under a concentrated force.

    AISC 360-16, Eqs. J10-2 and J10-3, Sect. J10.2, p. 16.1-143::

        distance from the member end > d:   Rn = Fyw*tw*(5*k + lb)  (J10-2)
        distance from the member end <= d:  Rn = Fyw*tw*(2.5*k + lb)(J10-3)

    with ``phi = 1.00`` / ``Omega = 1.50``.

    The near-end form halves the ``k`` term because the 2.5:1 load spread only
    develops on one side of the load when there is no member beyond it. ``lb``
    is unaffected -- the bearing length itself is still fully engaged.

    Sect. J10.2 notes that ``lb`` shall be "not less than k for end beam
    reactions".
    """
    if min(Fyw, tw, k) <= 0.0 or lb < 0.0:
        raise GeometryError(
            f"Fyw, tw and k must be positive and lb non-negative, got "
            f"Fyw={Fyw}, tw={tw}, k={k}, lb={lb}"
        )

    if near_member_end:
        Rn = Fyw * tw * (2.5 * k + lb)  # Eq. J10-3
        citation, note = cite("J10-3"), "force applied within d of the member end"
    else:
        Rn = Fyw * tw * (5.0 * k + lb)  # Eq. J10-2
        citation, note = cite("J10-2"), "force applied beyond d from the member end"

    state = LimitStateResult(
        LimitState.WEB_LOCAL_YIELDING, Rn, citation,
        detail={"Fyw": Fyw, "tw": tw, "k": k, "lb": lb}, note=note,
        phi=PHI_SHEAR_YIELDING, omega=OMEGA_SHEAR_YIELDING,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_SHEAR_YIELDING, omega=OMEGA_SHEAR_YIELDING, basis=basis
    )


def web_local_crippling(
    tw: Inch, tf: Inch, d: Inch, lb: Inch, E: Ksi, Fyw: Ksi, *, Qf: float = 1.0,
    near_member_end: bool = False, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Web local crippling under a compressive concentrated force.

    AISC 360-16, Eqs. J10-4, J10-5a and J10-5b, Sect. J10.3, p. 16.1-143::

        beyond d/2 from the end:
            Rn = 0.80*tw^2*[1 + 3*(lb/d)*(tw/tf)^1.5]
                 *sqrt(E*Fyw*tf/tw)*Qf                             (J10-4)

        within d/2, lb/d <= 0.2:
            Rn = 0.40*tw^2*[1 + 3*(lb/d)*(tw/tf)^1.5]
                 *sqrt(E*Fyw*tf/tw)*Qf                             (J10-5a)

        within d/2, lb/d > 0.2:
            Rn = 0.40*tw^2*[1 + (4*lb/d - 0.2)*(tw/tf)^1.5]
                 *sqrt(E*Fyw*tf/tw)*Qf                             (J10-5b)

    with ``phi = 0.75`` / ``Omega = 2.00``.

    Eq. J10-5a is exactly half of Eq. J10-4 -- an end reaction gets no help from
    web beyond the load. Eqs. J10-5a and J10-5b meet at ``lb/d = 0.2``, where
    ``3*(0.2) = 0.6`` and ``4*(0.2) - 0.2 = 0.6`` exactly.

    ``Qf`` is 1.0 for wide-flange sections and for HSS with the connecting
    surface in tension, and comes from Table K3.2 otherwise.
    """
    if min(tw, tf, d, E, Fyw) <= 0.0 or lb < 0.0:
        raise GeometryError("tw, tf, d, E and Fyw must be positive and lb non-negative")

    ratio = lb / d
    root = math.sqrt(E * Fyw * tf / tw)
    thickness_term = (tw / tf) ** 1.5

    if not near_member_end:
        bracket = 1.0 + 3.0 * ratio * thickness_term
        Rn = 0.80 * tw**2 * bracket * root * Qf  # Eq. J10-4
        citation, note = cite("J10-4"), "force applied beyond d/2 from the member end"
    elif ratio <= 0.2:
        bracket = 1.0 + 3.0 * ratio * thickness_term
        Rn = 0.40 * tw**2 * bracket * root * Qf  # Eq. J10-5a
        citation, note = cite("J10-5a"), "within d/2 of the end, lb/d <= 0.2"
    else:
        bracket = 1.0 + (4.0 * ratio - 0.2) * thickness_term
        Rn = 0.40 * tw**2 * bracket * root * Qf  # Eq. J10-5b
        citation, note = cite("J10-5b"), "within d/2 of the end, lb/d > 0.2"

    state = LimitStateResult(
        LimitState.WEB_LOCAL_CRIPPLING, Rn, citation,
        detail={"lb/d": ratio, "tw/tf": tw / tf, "Qf": Qf}, note=note,
        phi=PHI_RUPTURE, omega=OMEGA_RUPTURE,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_RUPTURE, omega=OMEGA_RUPTURE, basis=basis
    )


def web_sidesway_buckling(
    Cr: Ksi, tw: Inch, tf: Inch, h: Inch, Lb: Inch, bf: Inch, *,
    flange_restrained_against_rotation: bool = True, basis: Basis = Basis.LRFD,
) -> StrengthResult | None:
    """Web sidesway buckling under a compressive concentrated force.

    AISC 360-16, Eqs. J10-6 and J10-7, Sect. J10.4, p. 16.1-144::

        compression flange restrained, (h/tw)/(Lb/bf) <= 2.3:
            Rn = (Cr*tw^3*tf/h^2)*[1 + 0.4*((h/tw)/(Lb/bf))^3]     (J10-6)

        compression flange NOT restrained, (h/tw)/(Lb/bf) <= 1.7:
            Rn = (Cr*tw^3*tf/h^2)*[0.4*((h/tw)/(Lb/bf))^3]         (J10-7)

    with ``phi = 0.85`` / ``Omega = 1.76``.

    Returns ``None`` above the applicable ratio, where Sect. J10.4 states the
    limit state "does not apply" -- **not** a strength of zero. The difference
    matters: a member above the threshold is safe against this mode, not
    infinitely weak in it.

    Note that Eq. J10-7 lacks the leading ``1 +`` of Eq. J10-6. An unrestrained
    flange gets *only* the cubic term, which is why its threshold is lower.

    ``Cr`` is 960,000 ksi when ``Mu < My`` at the location of the force, and
    480,000 ksi when ``Mu >= My`` -- a factor of two, decided by whether the
    flange has already yielded.
    """
    if min(tw, tf, h, Lb, bf) <= 0.0:
        raise GeometryError("tw, tf, h, Lb and bf must be positive")

    ratio = (h / tw) / (Lb / bf)
    threshold = 2.3 if flange_restrained_against_rotation else 1.7
    if ratio > threshold:
        return None

    base = Cr * tw**3 * tf / h**2
    if flange_restrained_against_rotation:
        Rn = base * (1.0 + 0.4 * ratio**3)  # Eq. J10-6
        citation, note = cite("J10-6"), "compression flange restrained against rotation"
    else:
        Rn = base * (0.4 * ratio**3)  # Eq. J10-7
        citation, note = cite("J10-7"), "compression flange NOT restrained against rotation"

    state = LimitStateResult(
        LimitState.WEB_SIDESWAY_BUCKLING, Rn, citation,
        detail={"(h/tw)/(Lb/bf)": ratio, "Cr": Cr}, note=note, phi=0.85, omega=1.76,
    )
    return StrengthResult.build("Rn", "kip", [state], phi=0.85, omega=1.76, basis=basis)


def web_compression_buckling(
    tw: Inch, h: Inch, E: Ksi, Fyw: Ksi, *, Qf: float = 1.0, near_member_end: bool = False,
    basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Web compression buckling under a pair of concentrated forces.

    AISC 360-16, Eq. J10-8, Sect. J10.5, p. 16.1-145::

        Rn = (24*tw^3*sqrt(E*Fyw)/h)*Qf

    with ``phi = 0.90`` / ``Omega = 1.67``.

    Applies to a **pair** of compressive forces applied at both flanges at the
    same location -- a single force is web local crippling (Sect. J10.3)
    instead. When the pair is applied within ``d/2`` of the member end, ``Rn``
    is reduced by 50%.
    """
    if min(tw, h, E, Fyw) <= 0.0:
        raise GeometryError("tw, h, E and Fyw must be positive")

    Rn = (24.0 * tw**3 * math.sqrt(E * Fyw) / h) * Qf  # Eq. J10-8
    note = "a PAIR of forces at both flanges; a single force is Sect. J10.3 instead"
    if near_member_end:
        Rn *= 0.5
        note = "reduced 50%: applied within d/2 of the member end; " + note

    state = LimitStateResult(
        LimitState.WEB_COMPRESSION_BUCKLING, Rn, cite("J10-8"),
        detail={"tw": tw, "h": h, "Qf": Qf}, note=note,
        phi=PHI_YIELDING, omega=OMEGA_YIELDING,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_YIELDING, omega=OMEGA_YIELDING, basis=basis
    )


def panel_zone_shear(
    Fy: Ksi, dc: Inch, tw: Inch, Pr: Kip, Py: Kip, *, bcf: Inch = 0.0, tcf: Inch = 0.0,
    db: Inch = 0.0, inelastic_deformation_considered: bool = False,
    basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Web panel-zone shear yielding under double-concentrated forces.

    AISC 360-16, Eqs. J10-9 through J10-12, Sect. J10.6, pp. 16.1-145 to
    16.1-146, with ``phi = 0.90`` / ``Omega = 1.67``.

    When inelastic panel-zone deformation is **not** accounted for in the
    analysis::

        alpha*Pr <= 0.4*Py:  Rn = 0.60*Fy*dc*tw                    (J10-9)
        alpha*Pr >  0.4*Py:  Rn = 0.60*Fy*dc*tw*(1.4 - alpha*Pr/Py)(J10-10)

    When it **is** accounted for -- which means the frame analysis models the
    panel zone's post-yield stiffness -- the column flanges are credited::

        alpha*Pr <= 0.75*Py:
            Rn = 0.60*Fy*dc*tw*(1 + 3*bcf*tcf^2/(db*dc*tw))        (J10-11)
        alpha*Pr >  0.75*Py:
            Rn = 0.60*Fy*dc*tw*(1 + 3*bcf*tcf^2/(db*dc*tw))
                 *(1.9 - 1.2*alpha*Pr/Py)                          (J10-12)

    The flange term can add 20-40% for a heavy column, but taking it without
    modelling the inelastic panel zone is unconservative -- the strength it
    represents is only reached after the web has yielded and redistributed.

    Note Eq. J10-10 reaches zero at ``alpha*Pr = 1.4*Py``, which cannot occur,
    and Eq. J10-12 at ``1.583*Py``; both are safely outside the physical range.
    """
    if min(Fy, dc, tw) <= 0.0 or Py <= 0.0:
        raise GeometryError("Fy, dc, tw and Py must be positive")

    alpha = 1.0 if basis.is_lrfd else 1.6
    demand = alpha * abs(Pr) / Py
    base = 0.60 * Fy * dc * tw

    if not inelastic_deformation_considered:
        if demand <= 0.4:
            Rn, citation = base, cite("J10-9")
            note = "alpha*Pr <= 0.4*Py; inelastic panel-zone deformation not modelled"
        else:
            Rn, citation = base * (1.4 - demand), cite("J10-10")
            note = "alpha*Pr > 0.4*Py; axial load reduces the panel-zone shear strength"
    else:
        if min(bcf, tcf, db) <= 0.0:
            raise GeometryError(
                "Eqs. J10-11/J10-12 credit the column flanges and need bcf, tcf and db"
            )
        flange_term = 1.0 + 3.0 * bcf * tcf**2 / (db * dc * tw)
        if demand <= 0.75:
            Rn, citation = base * flange_term, cite("J10-11")
            note = "inelastic panel-zone deformation IS modelled; flanges credited"
        else:
            Rn, citation = base * flange_term * (1.9 - 1.2 * demand), cite("J10-12")
            note = "inelastic deformation modelled, alpha*Pr > 0.75*Py"

    state = LimitStateResult(
        LimitState.SHEAR_YIELDING, max(Rn, 0.0), citation,
        detail={"alpha*Pr/Py": demand, "0.60*Fy*dc*tw": base}, note=note,
        phi=PHI_YIELDING, omega=OMEGA_YIELDING,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_YIELDING, omega=OMEGA_YIELDING, basis=basis
    )
