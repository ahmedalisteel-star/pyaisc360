"""Chapter E -- Design of Members for Compression.

Covers Sects. E1 through E7, pp. 16.1-33 to 16.1-43.

The chapter's structure is a single idea applied four ways: find the elastic
buckling stress ``Fe`` for the relevant mode, convert it to a critical stress
``Fcr`` through Eq. E3-2/E3-3, and multiply by an area. What changes between
sections is *which* ``Fe`` and *which* area:

===========  ==========================================  ====================
Section      Elastic buckling stress                     Area
===========  ==========================================  ====================
E3           Eq. E3-4, flexural                          ``Ag``
E4           Eqs. E4-2/E4-3/E4-4, torsional and F-T       ``Ag``
E5           Eq. E3-4 with a modified ``Lc/r``           ``Ag`` (or ``Ae``)
E6           as above with ``(Lc/r)m`` from Eq. E6-1/2   ``Ag`` (or ``Ae``)
E7           as E3 or E4, unchanged                      ``Ae`` (Eqs. E7-2..7)
===========  ==========================================  ====================

That is why every ``Fe`` here is a standalone function: E5, E6 and E7 all reuse
E3's and E4's machinery rather than restating it.

The 2016 edition replaced the old ``Q``-factor treatment of slender elements
with the effective-width method of Sect. E7. ``Fcr`` is computed **once**, from
the full yield stress with no ``Q`` reduction, and the slenderness penalty is
taken entirely on the area. Applying a ``Q`` here as well would double-count.
"""

from __future__ import annotations

import cmath
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .chapter_b import AxialElement, limiting_width_to_thickness
from .core.citations import cite
from .core.config import DEFAULT_SETTINGS, Basis, DesignSettings
from .core.enums import LimitState
from .core.exceptions import AISC360Error, ConvergenceError, GeometryError, OutOfScopeError
from .core.result import LimitStateResult, StrengthResult
from .core.units import Inch, Inch2, Inch4, Inch6, Ksi, Ratio
from .materials import Steel
from .sections import require_properties
from .utils import elastic_buckling_stress, flexural_buckling_stress, limiting_ratio

__all__ = [
    "PHI_C",
    "OMEGA_C",
    "SLENDERNESS_ADVISORY_LIMIT",
    "Symmetry",
    "ConnectorType",
    "BuiltUpShape",
    "TrussType",
    "ConnectedLeg",
    "SlenderElement",
    "AngleConfig",
    "BuiltUpConfig",
    "CompressionMember",
    "effective_length",
    "flexural_buckling_stress_E3",
    "torsional_buckling_stress",
    "flexural_torsional_buckling_stress",
    "unsymmetric_buckling_stress",
    "Fex",
    "Fey",
    "Fez",
    "flexural_constant",
    "polar_radius_of_gyration",
    "single_angle_effective_slenderness",
    "built_up_modified_slenderness",
    "built_up_connector_spacing_limit",
    "c2_from_c1",
    "elastic_local_buckling_stress",
    "effective_width",
    "round_hss_effective_area",
    "effective_area",
    "compressive_strength",
]

#: Resistance and safety factors for compression.
#: AISC 360-16, Sect. E1, p. 16.1-33.
PHI_C: float = 0.90
OMEGA_C: float = 1.67

#: Sect. E2 User Note, p. 16.1-35: Lc/r "preferably should not exceed 200".
#: Advisory, so it is reported rather than enforced.
SLENDERNESS_ADVISORY_LIMIT: Ratio = 200.0


# ===========================================================================
# Configuration types
# ===========================================================================
class Symmetry(str, Enum):
    """Cross-section symmetry -- selects which Sect. E4 branch applies.

    ``SINGLY_X`` exists because of the User Note to Eq. E4-3, p. 16.1-36: for a
    section symmetric about the x-axis (a channel), Eq. E4-3 applies with
    ``Fey`` replaced by ``Fex``.
    """

    DOUBLY = "doubly symmetric"
    SINGLY_Y = "singly symmetric about the y-axis"
    SINGLY_X = "singly symmetric about the x-axis"
    UNSYMMETRIC = "unsymmetric"


class ConnectorType(str, Enum):
    """Intermediate connector type for a built-up member, Sect. E6.1."""

    SNUG_TIGHT_BOLTED = "snug-tight bolted"
    WELDED_OR_PRETENSIONED = "welded or pretensioned bolted"


class BuiltUpShape(str, Enum):
    """Built-up configuration, selecting ``Ki`` in Eq. E6-2b, p. 16.1-40."""

    ANGLES_BACK_TO_BACK = "angles back-to-back"
    CHANNELS_BACK_TO_BACK = "channels back-to-back"
    OTHER = "all other cases"

    @property
    def Ki(self) -> float:
        """Effective length factor for the built-up member. Sect. E6.1."""
        return {
            BuiltUpShape.ANGLES_BACK_TO_BACK: 0.50,
            BuiltUpShape.CHANNELS_BACK_TO_BACK: 0.75,
            BuiltUpShape.OTHER: 0.86,
        }[self]


class TrussType(str, Enum):
    """Which half of Sect. E5 applies, p. 16.1-38."""

    #: E5(a) -- individual members, or web members of planar trusses.
    PLANAR = "individual member or planar truss web member"
    #: E5(b) -- web members of box or space trusses.
    BOX_OR_SPACE = "box or space truss web member"


class ConnectedLeg(str, Enum):
    """Which leg of an angle carries the connection, Sect. E5."""

    EQUAL_OR_LONGER = "equal legs, or connected through the longer leg"
    SHORTER = "connected through the shorter leg"


@dataclass(frozen=True, slots=True)
class SlenderElement:
    """One compression element of a cross-section, for Sect. E7.

    Attributes
    ----------
    name:
        Label for the report, e.g. ``"flange"``.
    kind:
        Table B4.1a case, which fixes ``lambda_r`` and the Table E7.1 factors.
    b:
        Width of the element as Sect. B4.1 defines it: half the flange width
        for an I-shape flange, the clear web depth for a web, the full leg for
        an angle. **Not** the gross dimension.
    t:
        Thickness of the element.
    count:
        How many identical elements the section has -- 2 flanges, 1 web, 4 HSS
        walls. The area deduction is multiplied by this.
    lam:
        Width-to-thickness ratio. Defaults to ``b/t``; supply it explicitly to
        use a catalog value such as ``bf/2tf`` that may be rounded differently.
    """

    name: str
    kind: AxialElement
    b: Inch
    t: Inch
    count: int = 1
    lam: Ratio | None = None

    def __post_init__(self) -> None:
        if self.b <= 0.0 or self.t <= 0.0:
            raise GeometryError(f"{self.name}: b and t must be positive, got {self.b}, {self.t}")
        if self.count < 1:
            raise GeometryError(f"{self.name}: count must be at least 1, got {self.count}")

    @property
    def ratio(self) -> Ratio:
        """Width-to-thickness ratio actually used."""
        return self.lam if self.lam is not None else self.b / self.t


@dataclass(frozen=True, slots=True)
class AngleConfig:
    """Single-angle configuration for Sect. E5, p. 16.1-38.

    Attributes
    ----------
    L:
        Length of the member between work points at the truss chord
        centrelines, in. Note this is ``L``, not ``Lc``: Sect. E5 derives the
        effective slenderness from the *unmodified* geometric length.
    ra:
        Radius of gyration about the geometric axis parallel to the connected
        leg, in.
    rz:
        Radius of gyration about the minor principal axis, in.
    b_long, b_short:
        Long and short leg widths, in. Equal for an equal-leg angle.
    truss:
        Whether Sect. E5(a) or E5(b) applies.
    connected_leg:
        Which leg is connected.
    """

    L: Inch
    ra: Inch
    rz: Inch
    b_long: Inch
    b_short: Inch
    truss: TrussType = TrussType.PLANAR
    connected_leg: ConnectedLeg = ConnectedLeg.EQUAL_OR_LONGER

    def __post_init__(self) -> None:
        if min(self.L, self.ra, self.rz, self.b_long, self.b_short) <= 0.0:
            raise GeometryError("all AngleConfig dimensions must be positive")
        if self.b_short > self.b_long:
            raise GeometryError(
                f"b_long ({self.b_long}) must be at least b_short ({self.b_short}); "
                "the arguments look transposed"
            )


@dataclass(frozen=True, slots=True)
class BuiltUpConfig:
    """Built-up member configuration for Sect. E6, p. 16.1-39.

    Attributes
    ----------
    a:
        Distance between intermediate connectors, in.
    ri:
        Minimum radius of gyration of an individual component, in.
    connector:
        Snug-tight bolted, or welded/pretensioned.
    shape:
        Selects ``Ki`` for Eq. E6-2b.
    """

    a: Inch
    ri: Inch
    connector: ConnectorType = ConnectorType.WELDED_OR_PRETENSIONED
    shape: BuiltUpShape = BuiltUpShape.OTHER

    def __post_init__(self) -> None:
        if self.a <= 0.0 or self.ri <= 0.0:
            raise GeometryError(f"a and ri must be positive, got a={self.a}, ri={self.ri}")


@dataclass(frozen=True)
class CompressionMember:
    """A member in axial compression.

    Attributes
    ----------
    section:
        Anything satisfying the Wave 0 section contract -- a
        :class:`~pyaisc360.sections.SectionAdapter` or an object already using
        the canonical names.
    steel:
        Grade, supplying ``Fy``, ``E`` and ``G``.
    Lcx, Lcy:
        Effective lengths for flexural buckling about each axis, in.
        (Sect. E2: ``Lc = KL``.)
    Lcz:
        Effective length for torsional buckling about the longitudinal axis,
        in. Defaults to ``Lcy``.
    symmetry:
        Which Sect. E4 branch applies.
    xo, yo:
        Shear-centre coordinates with respect to the centroid, in. Required
        only for the singly symmetric and unsymmetric branches of Sect. E4.
    include_warping:
        Set False for tees and double angles: the User Note on p. 16.1-37
        directs that the ``Cw`` term be omitted from ``Fez`` and ``xo`` taken
        as zero for those shapes.
    """

    section: Any
    steel: Steel
    Lcx: Inch
    Lcy: Inch
    Lcz: Inch | None = None
    symmetry: Symmetry = Symmetry.DOUBLY
    xo: Inch = 0.0
    yo: Inch = 0.0
    include_warping: bool = True
    elements: Sequence[SlenderElement] = field(default_factory=tuple)
    angle: AngleConfig | None = None
    built_up: BuiltUpConfig | None = None

    def __post_init__(self) -> None:
        if self.Lcx < 0.0 or self.Lcy < 0.0:
            raise GeometryError(
                f"effective lengths must be non-negative, got {self.Lcx}, {self.Lcy}"
            )

    @property
    def Lcz_effective(self) -> Inch:
        return self.Lcy if self.Lcz is None else self.Lcz


# ===========================================================================
# Sect. E2 -- Effective Length
# ===========================================================================
def effective_length(K: float, L: Inch) -> Inch:
    """Effective length ``Lc = KL``.

    AISC 360-16, Sect. E2, p. 16.1-35.

    Sect. E2 also permits ``Lc`` to be established by an elastic buckling
    analysis rather than through ``K``; this helper covers only the ``K``-factor
    route, which is why it is a trivial product rather than the sole entry point.
    """
    if K <= 0.0:
        raise GeometryError(f"effective length factor K must be positive, got {K}")
    if L < 0.0:
        raise GeometryError(f"unbraced length must be non-negative, got {L}")
    return K * L


# ===========================================================================
# Sect. E3 -- Flexural Buckling of Members without Slender Elements
# ===========================================================================
def flexural_buckling_stress_E3(slenderness: Ratio, E: Ksi, Fy: Ksi) -> tuple[Ksi, Ksi]:
    """Critical and elastic stresses for flexural buckling.

    AISC 360-16, Sect. E3, p. 16.1-35. Returns ``(Fcr, Fe)``.

    ``Fe`` from Eq. E3-4; ``Fcr`` from Eq. E3-2 or Eq. E3-3 depending on
    ``Fy/Fe``. Both are returned because a calculation sheet shows both.
    """
    Fe = elastic_buckling_stress(slenderness, E)  # Eq. E3-4
    return flexural_buckling_stress(Fy, Fe), Fe  # Eqs. E3-2 / E3-3


# ===========================================================================
# Sect. E4 -- Torsional and Flexural-Torsional Buckling
# ===========================================================================
def Fex(Lcx_over_rx: Ratio, E: Ksi) -> Ksi:
    """Elastic flexural buckling stress about the x-axis.

    AISC 360-16, Eq. E4-5, Sect. E4, p. 16.1-36::

        Fex = pi^2*E / (Lcx/rx)^2
    """
    return elastic_buckling_stress(Lcx_over_rx, E)


def Fey(Lcy_over_ry: Ratio, E: Ksi) -> Ksi:
    """Elastic flexural buckling stress about the y-axis.

    AISC 360-16, Eq. E4-6, Sect. E4, p. 16.1-36::

        Fey = pi^2*E / (Lcy/ry)^2
    """
    return elastic_buckling_stress(Lcy_over_ry, E)


def Fez(
    E: Ksi,
    G: Ksi,
    Cw: Inch6,
    J: Inch4,
    Lcz: Inch,
    Ag: Inch2,
    ro: Inch,
    *,
    include_warping: bool = True,
) -> Ksi:
    """Elastic torsional buckling stress about the longitudinal axis.

    AISC 360-16, Eq. E4-7, Sect. E4, p. 16.1-36::

        Fez = (pi^2*E*Cw/Lcz^2 + G*J) * 1/(Ag*ro^2)

    Parameters
    ----------
    include_warping:
        False for tees and double angles. The User Note on p. 16.1-37 directs
        that the ``Cw`` term be omitted for those shapes, because their warping
        resistance is negligible and the tabulated ``Cw`` is not a meaningful
        measure of it.
    """
    if Lcz <= 0.0:
        raise GeometryError(f"Lcz must be positive, got {Lcz}")
    if Ag <= 0.0 or ro <= 0.0:
        raise GeometryError(f"Ag and ro must be positive, got Ag={Ag}, ro={ro}")
    warping = math.pi**2 * E * Cw / Lcz**2 if include_warping else 0.0
    return (warping + G * J) / (Ag * ro**2)


def polar_radius_of_gyration(xo: Inch, yo: Inch, Ix: Inch4, Iy: Inch4, Ag: Inch2) -> Inch:
    """Polar radius of gyration about the shear centre, ``ro``.

    AISC 360-16, Eq. E4-9, Sect. E4, p. 16.1-37::

        ro^2 = xo^2 + yo^2 + (Ix + Iy)/Ag
    """
    if Ag <= 0.0:
        raise GeometryError(f"Ag must be positive, got {Ag}")
    if Ix <= 0.0 or Iy <= 0.0:
        raise GeometryError(f"Ix and Iy must be positive, got {Ix}, {Iy}")
    return math.sqrt(xo**2 + yo**2 + (Ix + Iy) / Ag)


def flexural_constant(xo: Inch, yo: Inch, ro: Inch) -> Ratio:
    """Flexural constant ``H``.

    AISC 360-16, Eq. E4-8, Sect. E4, p. 16.1-36::

        H = 1 - (xo^2 + yo^2)/ro^2

    ``H`` approaches zero as the shear centre moves far from the centroid,
    at which point Eq. E4-3 becomes numerically singular -- guarded here.
    """
    if ro <= 0.0:
        raise GeometryError(f"ro must be positive, got {ro}")
    H = 1.0 - (xo**2 + yo**2) / ro**2
    if H <= 0.0:
        raise GeometryError(
            f"flexural constant H = {H:.4g} is not positive; the shear-centre "
            "offset exceeds the polar radius of gyration, which is geometrically "
            "impossible -- check xo, yo, Ix, Iy and Ag"
        )
    return H


def torsional_buckling_stress(
    E: Ksi, G: Ksi, Cw: Inch6, J: Inch4, Lcz: Inch, Ix: Inch4, Iy: Inch4
) -> Ksi:
    """Elastic torsional buckling stress for a **doubly symmetric** member.

    AISC 360-16, Eq. E4-2, Sect. E4, p. 16.1-36::

        Fe = (pi^2*E*Cw/Lcz^2 + G*J) * 1/(Ix + Iy)

    Identical to Eq. E4-7 for a doubly symmetric section, because ``xo = yo = 0``
    there makes ``Ag*ro^2 = Ix + Iy`` exactly. The two are kept as separate
    functions because the Specification does, and because Eq. E4-7 is also used
    inside Eqs. E4-3 and E4-4 where the identity does not hold.
    """
    if Lcz <= 0.0:
        raise GeometryError(f"Lcz must be positive, got {Lcz}")
    if Ix <= 0.0 or Iy <= 0.0:
        raise GeometryError(f"Ix and Iy must be positive, got {Ix}, {Iy}")
    return (math.pi**2 * E * Cw / Lcz**2 + G * J) / (Ix + Iy)


def flexural_torsional_buckling_stress(F_flexural: Ksi, F_torsional: Ksi, H: Ratio) -> Ksi:
    """Elastic flexural-torsional buckling stress for a **singly symmetric** member.

    AISC 360-16, Eq. E4-3, Sect. E4, p. 16.1-36::

        Fe = ((Fey + Fez)/(2H)) * [1 - sqrt(1 - 4*Fey*Fez*H/(Fey + Fez)^2)]

    Parameters
    ----------
    F_flexural:
        ``Fey`` when the **y**-axis is the axis of symmetry. Per the User Note
        on p. 16.1-36, pass ``Fex`` instead when the **x**-axis is the axis of
        symmetry, as for a channel.
    F_torsional:
        ``Fez`` from Eq. E4-7.
    H:
        Flexural constant from Eq. E4-8.

    Notes
    -----
    The radicand is non-negative for any real section, because
    ``4*Fey*Fez*H <= (Fey + Fez)^2`` follows from ``H <= 1`` and the AM-GM
    inequality. It is clamped at zero anyway so that floating-point noise at
    ``H = 1`` with ``Fey == Fez`` cannot produce a domain error.
    """
    if F_flexural <= 0.0 or F_torsional <= 0.0:
        raise GeometryError(
            f"component buckling stresses must be positive, got {F_flexural}, {F_torsional}"
        )
    if not 0.0 < H <= 1.0:
        raise GeometryError(f"flexural constant H must lie in (0, 1], got {H}")

    total = F_flexural + F_torsional
    radicand = 1.0 - 4.0 * F_flexural * F_torsional * H / total**2
    return (total / (2.0 * H)) * (1.0 - math.sqrt(max(radicand, 0.0)))


def unsymmetric_buckling_stress(
    F_ex: Ksi, F_ey: Ksi, F_ez: Ksi, xo: Inch, yo: Inch, ro: Inch
) -> Ksi:
    """Elastic buckling stress for an **unsymmetric** member: the Eq. E4-4 cubic.

    AISC 360-16, Eq. E4-4, Sect. E4, p. 16.1-36. ``Fe`` is the lowest root of::

        (Fe - Fex)(Fe - Fey)(Fe - Fez)
            - Fe^2*(Fe - Fey)*(xo/ro)^2
            - Fe^2*(Fe - Fex)*(yo/ro)^2 = 0

    Solved exactly, not iteratively. Expanding and collecting powers of ``Fe``
    gives, with ``H = 1 - (xo^2 + yo^2)/ro^2`` from Eq. E4-8::

        H*Fe^3
          + [(xo/ro)^2*Fey + (yo/ro)^2*Fex - (Fex + Fey + Fez)]*Fe^2
          + (Fex*Fey + Fey*Fez + Fez*Fex)*Fe
          - Fex*Fey*Fez = 0

    The cubic coefficient is ``H`` exactly, because the two subtracted terms
    contribute ``-(xo^2 + yo^2)/ro^2`` to the leading coefficient of 1. That is
    also why ``H -> 0`` degenerates the cubic to a quadratic, and why
    :func:`flexural_constant` refuses a non-positive ``H``.

    The depressed cubic is then solved in closed form -- the trigonometric
    (Viete) form when all three roots are real, Cardano otherwise. For any real
    cross-section all three roots are real and positive, so the trigonometric
    branch is the normal path.

    Raises
    ------
    ConvergenceError
        If no positive real root exists, which means the inputs are not a
        physically realisable section.
    """
    if min(F_ex, F_ey, F_ez) <= 0.0:
        raise GeometryError(
            f"component buckling stresses must be positive, got {F_ex}, {F_ey}, {F_ez}"
        )
    if ro <= 0.0:
        raise GeometryError(f"ro must be positive, got {ro}")

    alpha = (xo / ro) ** 2
    beta = (yo / ro) ** 2
    H = 1.0 - alpha - beta
    if H <= 0.0:
        raise GeometryError(
            f"flexural constant H = {H:.4g} is not positive; check xo, yo and ro"
        )

    # Coefficients of H*Fe^3 + a2*Fe^2 + a1*Fe + a0 = 0, normalised by H.
    a2 = (alpha * F_ey + beta * F_ex - (F_ex + F_ey + F_ez)) / H
    a1 = (F_ex * F_ey + F_ey * F_ez + F_ez * F_ex) / H
    a0 = -(F_ex * F_ey * F_ez) / H

    roots = _real_cubic_roots(a2, a1, a0)
    positive = [r for r in roots if r > 0.0]
    if not positive:
        raise ConvergenceError(
            f"Eq. E4-4 has no positive real root for Fex={F_ex:g}, Fey={F_ey:g}, "
            f"Fez={F_ez:g}, xo={xo:g}, yo={yo:g}, ro={ro:g}"
        )
    return min(positive)


def _real_cubic_roots(a2: float, a1: float, a0: float) -> list[float]:
    """Real roots of ``x^3 + a2*x^2 + a1*x + a0 = 0``, in closed form.

    Substituting ``x = t - a2/3`` gives the depressed cubic ``t^3 + p*t + q = 0``.
    When the discriminant is non-positive there are three real roots, recovered
    by the trigonometric method; otherwise there is one, recovered by Cardano.
    Both are exact -- no iteration, no tolerance.
    """
    shift = a2 / 3.0
    p = a1 - a2**2 / 3.0
    q = 2.0 * a2**3 / 27.0 - a2 * a1 / 3.0 + a0

    if abs(p) < 1e-300:  # t^3 = -q
        return [math.copysign(abs(q) ** (1.0 / 3.0), -q) - shift]

    discriminant = (q / 2.0) ** 2 + (p / 3.0) ** 3

    # A repeated root puts the discriminant at exactly zero, where rounding can
    # push it either way. Landing on the Cardano branch there would return one
    # root and silently drop the repeated pair, so the near-zero band is scaled
    # against the terms it came from and routed to the three-real-root branch.
    tolerance = 1e-12 * max(1.0, (q / 2.0) ** 2, abs(p / 3.0) ** 3)

    if discriminant <= tolerance and p < 0.0:
        # Three real roots -- Viete's trigonometric solution.
        m = 2.0 * math.sqrt(-p / 3.0)
        argument = 3.0 * q / (p * m)
        argument = min(max(argument, -1.0), 1.0)  # guard rounding at the cusp
        theta = math.acos(argument) / 3.0
        return [
            m * math.cos(theta - 2.0 * math.pi * k / 3.0) - shift for k in range(3)
        ]

    # One real root -- Cardano, via complex cube roots so the sign works out.
    u = (-q / 2.0 + cmath.sqrt(discriminant)) ** (1.0 / 3.0)
    if abs(u) < 1e-300:
        u = (-q / 2.0 - cmath.sqrt(discriminant)) ** (1.0 / 3.0)
    root = (u - p / (3.0 * u)).real - shift
    return [root]


# ===========================================================================
# Sect. E5 -- Single-Angle Compression Members
# ===========================================================================
def single_angle_effective_slenderness(config: AngleConfig, *, strict: bool = True) -> Ratio:
    """Effective slenderness ``Lc/r`` for a single angle.

    AISC 360-16, Sect. E5, Eqs. E5-1 through E5-4, pp. 16.1-38 to 16.1-39.

    E5(a), individual members and planar-truss web members::

        L/ra <= 80:  Lc/r = 72 + 0.75*L/ra          (Eq. E5-1)
        L/ra >  80:  Lc/r = 32 + 1.25*L/ra          (Eq. E5-2)

    E5(b), box and space truss web members::

        L/ra <= 75:  Lc/r = 60 + 0.80*L/ra          (Eq. E5-3)
        L/ra >  75:  Lc/r = 45 + 1.00*L/ra          (Eq. E5-4)

    For unequal-leg angles connected through the **shorter** leg, the result is
    increased by ``4*[(bl/bs)^2 - 1]`` under E5(a) or ``6*[(bl/bs)^2 - 1]``
    under E5(b), and then floored at ``0.95*L/rz`` or ``0.82*L/rz``
    respectively.

    These expressions absorb the connection eccentricity, which is why Sect. E5
    then permits the member to be treated as axially loaded. They are only valid
    inside the five conditions listed on p. 16.1-38; ``strict`` enforces the two
    that are checkable from geometry alone.

    Parameters
    ----------
    strict:
        Enforce ``Lc/r <= 200`` (condition 4) and, for unequal legs, the
        long-to-short ratio below 1.7 (condition 5). The other three conditions
        -- loaded through one leg at both ends, at least two bolts or welded, no
        intermediate transverse loads -- are detailing facts this function
        cannot see, and remain the engineer's to confirm.

    Raises
    ------
    OutOfScopeError
        When ``strict`` and a checkable condition of Sect. E5 is violated. A
        member failing any of them must be designed for combined axial force and
        flexure under Chapter H instead.
    """
    leg_ratio = config.b_long / config.b_short
    slenderness_L_ra = config.L / config.ra

    if config.truss is TrussType.PLANAR:
        if slenderness_L_ra <= 80.0:
            Lc_r = 72.0 + 0.75 * slenderness_L_ra  # Eq. E5-1
        else:
            Lc_r = 32.0 + 1.25 * slenderness_L_ra  # Eq. E5-2
        increment, floor_factor = 4.0, 0.95
    else:
        if slenderness_L_ra <= 75.0:
            Lc_r = 60.0 + 0.80 * slenderness_L_ra  # Eq. E5-3
        else:
            Lc_r = 45.0 + 1.00 * slenderness_L_ra  # Eq. E5-4
        increment, floor_factor = 6.0, 0.82

    if config.connected_leg is ConnectedLeg.SHORTER:
        if strict and leg_ratio >= 1.7:
            raise OutOfScopeError(
                f"Sect. E5 condition (5), p. 16.1-38: the long-to-short leg ratio "
                f"is {leg_ratio:.3f}, which is not less than 1.7. Design this angle "
                "for combined axial force and flexure under Chapter H."
            )
        Lc_r += increment * (leg_ratio**2 - 1.0)
        Lc_r = max(Lc_r, floor_factor * config.L / config.rz)

    if strict and Lc_r > SLENDERNESS_ADVISORY_LIMIT:
        raise OutOfScopeError(
            f"Sect. E5 condition (4), p. 16.1-38: Lc/r = {Lc_r:.1f} exceeds 200. "
            "The simplified single-angle provisions do not apply; use Chapter H."
        )
    return Lc_r


def single_angle_ftb_can_be_neglected(b_over_t: Ratio, E: Ksi, Fy: Ksi) -> bool:
    """Whether flexural-torsional buckling need be considered for a single angle.

    AISC 360-16, Sect. E5, p. 16.1-38: "Flexural-torsional buckling need not be
    considered when ``b/t <= 0.71*sqrt(E/Fy)``", where ``b`` is the width of the
    longest leg.
    """
    return b_over_t <= limiting_ratio(0.71, E, Fy)


# ===========================================================================
# Sect. E6 -- Built-Up Members
# ===========================================================================
def built_up_modified_slenderness(slenderness: Ratio, config: BuiltUpConfig) -> Ratio:
    """Modified slenderness ``(Lc/r)m`` for a built-up member.

    AISC 360-16, Sect. E6.1, Eqs. E6-1, E6-2a and E6-2b, pp. 16.1-39 to 16.1-40.

    (a) Snug-tight bolted intermediate connectors::

            (Lc/r)m = sqrt((Lc/r)o^2 + (a/ri)^2)            (Eq. E6-1)

    (b) Welded, or pretensioned bolted with Class A or B faying surfaces::

            a/ri <= 40:  (Lc/r)m = (Lc/r)o                  (Eq. E6-2a)
            a/ri >  40:  (Lc/r)m = sqrt((Lc/r)o^2 + (Ki*a/ri)^2)   (Eq. E6-2b)

    ``Ki`` is 0.50 for angles back-to-back, 0.75 for channels back-to-back and
    0.86 otherwise.

    The modification applies only when the buckling mode produces shear in the
    connectors -- buckling about the axis through which the shapes are joined
    does not, and takes the unmodified ratio. Sect. E6.1 makes that the caller's
    determination, so this function applies the modification unconditionally to
    whatever slenderness it is given.

    Parameters
    ----------
    slenderness:
        ``(Lc/r)o``, the slenderness of the built-up member acting as a unit in
        the buckling direction being addressed.
    """
    if slenderness <= 0.0:
        raise GeometryError(f"(Lc/r)o must be positive, got {slenderness}")

    a_over_ri = config.a / config.ri

    if config.connector is ConnectorType.SNUG_TIGHT_BOLTED:
        return math.sqrt(slenderness**2 + a_over_ri**2)  # Eq. E6-1

    if a_over_ri <= 40.0:
        return slenderness  # Eq. E6-2a
    return math.sqrt(slenderness**2 + (config.shape.Ki * a_over_ri) ** 2)  # Eq. E6-2b


def built_up_connector_spacing_limit(governing_slenderness: Ratio) -> Ratio:
    """Maximum ``a/ri`` between intermediate connectors.

    AISC 360-16, Sect. E6.2(a), p. 16.1-40: components shall be connected at
    intervals such that ``a/ri`` does not exceed three-fourths of the governing
    slenderness ratio of the built-up member, using the least radius of gyration
    of each component.

    This is a **dimensional requirement**, not a strength equation -- it has no
    equation number, and satisfying it does not change ``Pn``. It is checked
    separately by :func:`compressive_strength` and reported as a note.
    """
    if governing_slenderness <= 0.0:
        raise GeometryError(f"governing slenderness must be positive, got {governing_slenderness}")
    return 0.75 * governing_slenderness


# ===========================================================================
# Sect. E7 -- Members with Slender Elements
# ===========================================================================
#: Table E7.1, p. 16.1-43 -- effective width imperfection adjustment factors.
#: Only ``c1`` is independent; ``c2`` follows from Eq. E7-4 and is stored here
#: solely as the published cross-check that :func:`c2_from_c1` reproduces.
_TABLE_E7_1: dict[str, tuple[float, float]] = {
    "stiffened": (0.18, 1.31),  # case (a), except walls of square/rect HSS
    "hss_wall": (0.20, 1.38),  # case (b)
    "other": (0.22, 1.49),  # case (c), all other elements
}


def _e7_case(kind: AxialElement) -> str:
    """Which Table E7.1 row an element falls in, p. 16.1-43."""
    if kind is AxialElement.RECTANGULAR_HSS_WALL:
        return "hss_wall"
    return "stiffened" if kind.is_stiffened else "other"


def c2_from_c1(c1: float) -> float:
    """Second imperfection adjustment factor ``c2``.

    AISC 360-16, Eq. E7-4, Sect. E7.1, p. 16.1-43::

        c2 = (1 - sqrt(1 - 4*c1)) / (2*c1)

    Reproduces the ``c2`` column of Table E7.1 to the two decimals printed
    there, which is the check :mod:`tests.test_chapter_e` asserts.
    """
    if not 0.0 < c1 <= 0.25:
        raise AISC360Error(f"c1 must lie in (0, 0.25] for Eq. E7-4 to be real, got {c1}")
    return (1.0 - math.sqrt(1.0 - 4.0 * c1)) / (2.0 * c1)


def elastic_local_buckling_stress(lam: Ratio, lam_r: Ratio, c2: float, Fy: Ksi) -> Ksi:
    """Elastic local buckling stress ``Fel``.

    AISC 360-16, Eq. E7-5, Sect. E7.1, p. 16.1-43::

        Fel = (c2*lambda_r/lambda)^2 * Fy
    """
    if lam <= 0.0:
        raise GeometryError(f"lambda must be positive, got {lam}")
    if lam_r <= 0.0:
        raise AISC360Error(f"lambda_r must be positive, got {lam_r}")
    return (c2 * lam_r / lam) ** 2 * Fy


def effective_width(
    b: Inch, lam: Ratio, lam_r: Ratio, Fy: Ksi, Fcr: Ksi, c1: float, c2: float
) -> Inch:
    """Effective width ``be`` of a slender element.

    AISC 360-16, Eqs. E7-2 and E7-3, Sect. E7.1, p. 16.1-42::

        lambda <= lambda_r*sqrt(Fy/Fcr):  be = b                        (E7-2)
        lambda >  lambda_r*sqrt(Fy/Fcr):  be = b*(1 - c1*sqrt(Fel/Fcr))
                                                 *sqrt(Fel/Fcr)        (E7-3)

    For tees the element width is ``d`` and for webs it is ``h``; the caller
    supplies whichever applies as ``b``.

    Note the threshold is **not** ``lambda_r``. An element classified slender at
    ``lambda_r`` can still be fully effective once ``Fcr`` falls below ``Fy``,
    because a long column buckles before its plate elements do. Testing against
    ``lambda_r`` alone would deduct area from a slender column that has none to
    lose -- an unconservative-looking error that is in fact over-conservative,
    and wrong either way.

    The result is clamped to ``b``: Eq. E7-3 is a reduction, and rounding at the
    threshold must never return more width than the element has.
    """
    if b <= 0.0:
        raise GeometryError(f"element width must be positive, got {b}")
    if Fcr <= 0.0:
        raise AISC360Error(f"Fcr must be positive, got {Fcr}")

    if lam <= lam_r * math.sqrt(Fy / Fcr):
        return b  # Eq. E7-2

    Fel = elastic_local_buckling_stress(lam, lam_r, c2, Fy)  # Eq. E7-5
    ratio = math.sqrt(Fel / Fcr)
    be = b * (1.0 - c1 * ratio) * ratio  # Eq. E7-3
    return min(max(be, 0.0), b)


def round_hss_effective_area(Ag: Inch2, D: Inch, t: Inch, E: Ksi, Fy: Ksi) -> Inch2:
    """Effective area ``Ae`` of a round HSS.

    AISC 360-16, Eqs. E7-6 and E7-7, Sect. E7.2, p. 16.1-43::

        D/t <= 0.11*E/Fy:                    Ae = Ag                     (E7-6)
        0.11*E/Fy < D/t < 0.45*E/Fy:  Ae = [0.038*E/(Fy*(D/t)) + 2/3]*Ag (E7-7)

    Raises
    ------
    OutOfScopeError
        If ``D/t >= 0.45*E/Fy``. Sect. E7.2 gives no provision beyond that
        limit -- the Specification simply does not cover such a section, and
        extrapolating Eq. E7-7 past it is unconservative.
    """
    if Ag <= 0.0 or D <= 0.0 or t <= 0.0:
        raise GeometryError(f"Ag, D and t must be positive, got {Ag}, {D}, {t}")

    D_over_t = D / t
    lower = 0.11 * E / Fy
    upper = 0.45 * E / Fy

    if D_over_t <= lower:
        return Ag  # Eq. E7-6
    if D_over_t >= upper:
        raise OutOfScopeError(
            f"D/t = {D_over_t:.1f} reaches or exceeds 0.45*E/Fy = {upper:.1f}. "
            "AISC 360-16 Sect. E7.2, p. 16.1-43, provides no compressive strength "
            "for round HSS beyond this limit."
        )
    return (0.038 * E / (Fy * D_over_t) + 2.0 / 3.0) * Ag  # Eq. E7-7


def effective_area(
    Ag: Inch2, elements: Sequence[SlenderElement], E: Ksi, Fy: Ksi, Fcr: Ksi
) -> tuple[Inch2, dict[str, float]]:
    """Effective area ``Ae`` by deducting each slender element's lost width.

    AISC 360-16, Sect. E7, p. 16.1-42. Per the User Note there::

        Ae = Ag - sum over elements of (b - be)*t

    Returns ``(Ae, detail)``, where ``detail`` records each element's ratio and
    effective width for the calculation sheet.

    Working by deduction rather than by summing effective areas is the User
    Note's own method, and it is the safer one: it cannot lose the fillet and
    corner material that is part of ``Ag`` but belongs to no flat element.
    """
    if Ag <= 0.0:
        raise GeometryError(f"Ag must be positive, got {Ag}")

    Ae = Ag
    detail: dict[str, float] = {}
    for element in elements:
        lam_r = limiting_width_to_thickness(element.kind, E, Fy)
        c1, published_c2 = _TABLE_E7_1[_e7_case(element.kind)]
        c2 = c2_from_c1(c1)
        be = effective_width(element.b, element.ratio, lam_r, Fy, Fcr, c1, c2)
        Ae -= element.count * (element.b - be) * element.t
        detail[f"{element.name}: lambda"] = element.ratio
        detail[f"{element.name}: lambda_r"] = lam_r
        detail[f"{element.name}: be"] = be
        del published_c2

    if Ae <= 0.0:
        raise GeometryError(
            f"effective area came out non-positive ({Ae:.4g} in.^2). The element "
            "widths supplied exceed the gross area -- check that b is the element "
            "width per Sect. B4.1 (half the flange, the clear web depth) and not a "
            "gross dimension."
        )
    return Ae, detail


# ===========================================================================
# Sect. E1 -- the orchestrator
# ===========================================================================
def compressive_strength(
    member: CompressionMember,
    *,
    basis: Basis = Basis.LRFD,
    settings: DesignSettings = DEFAULT_SETTINGS,
) -> StrengthResult:
    """Nominal and available compressive strength, ``Pn``.

    AISC 360-16, Sect. E1, p. 16.1-33: "The nominal compressive strength, Pn,
    shall be the lowest value obtained based on the applicable limit states of
    flexural buckling, torsional buckling, and flexural-torsional buckling",
    with ``phi_c = 0.90`` (LRFD) and ``Omega_c = 1.67`` (ASD).

    Evaluates, as applicable:

    * flexural buckling about each axis (Sect. E3), with the slenderness
      modified by Sect. E5 for a single angle or Sect. E6 for a built-up member;
    * torsional or flexural-torsional buckling (Sect. E4), by the branch the
      member's symmetry selects;
    * the effective-area reduction of Sect. E7 when slender elements are given.

    Returns a :class:`~pyaisc360.core.result.StrengthResult` carrying every
    limit state evaluated, so the governing mode and the margin over the others
    are both visible.

    Notes
    -----
    When ``member.elements`` is empty the section is taken as having no slender
    elements, and the result says so. Supply the elements to have Sect. E7
    applied.
    """
    section = member.section
    steel = member.steel
    E, Fy, G = steel.E, steel.Fy, steel.G

    props = require_properties(section, "Ag", "rx", "ry")
    Ag, rx, ry = props["Ag"], props["rx"], props["ry"]

    notes: list[str] = []

    # -- slenderness, with the Sect. E5 / E6 modifications ------------------
    if member.angle is not None:
        Lc_r_governing = single_angle_effective_slenderness(member.angle, strict=settings.strict)
        slenderness_source = "Sect. E5 effective slenderness"
        slendernesses = {"E5": Lc_r_governing}
    else:
        slendernesses = {
            "x": member.Lcx / rx,
            "y": member.Lcy / ry,
        }
        if member.built_up is not None:
            slendernesses = {
                axis: built_up_modified_slenderness(value, member.built_up)
                for axis, value in slendernesses.items()
            }
            slenderness_source = "Sect. E6 modified slenderness (Lc/r)m"
        else:
            slenderness_source = "Lc/r"
        Lc_r_governing = max(slendernesses.values())

    if Lc_r_governing > SLENDERNESS_ADVISORY_LIMIT:
        notes.append(
            f"{slenderness_source} = {Lc_r_governing:.1f} exceeds the 200 the "
            "Sect. E2 User Note advises against (advisory, not a limit)"
        )

    if member.built_up is not None:
        limit = built_up_connector_spacing_limit(Lc_r_governing)
        actual = member.built_up.a / member.built_up.ri
        if actual > limit:
            notes.append(
                f"Sect. E6.2(a) NOT satisfied: a/ri = {actual:.1f} exceeds "
                f"0.75*(Lc/r) = {limit:.1f}; reduce the connector spacing"
            )

    # -- Sect. E3, flexural buckling ---------------------------------------
    limit_states: list[LimitStateResult | None] = []
    Fcr_flexural, Fe_flexural = flexural_buckling_stress_E3(Lc_r_governing, E, Fy)

    # -- Sect. E4, torsional / flexural-torsional --------------------------
    Fcr_torsional, torsional_detail, torsional_citation = _evaluate_E4(member, E, G, Ag, rx, ry)

    # -- Sect. E7, effective area ------------------------------------------
    governing_Fcr = min(
        [Fcr_flexural] + ([Fcr_torsional] if Fcr_torsional is not None else [])
    )
    if member.elements:
        Ae, area_detail = effective_area(Ag, member.elements, E, Fy, governing_Fcr)
        area_citation = "E7-1"
        if Ae < Ag:
            notes.append(
                f"Sect. E7 applies: Ae = {Ae:.3f} in.^2 vs Ag = {Ag:.3f} in.^2 "
                f"({100 * (1 - Ae / Ag):.1f}% reduction)"
            )
    else:
        Ae, area_detail = Ag, {}
        area_citation = "E3-1"
        notes.append(
            "no elements supplied -- section taken as nonslender per Sect. B4.1; "
            "pass elements= to apply Sect. E7"
        )

    limit_states.append(
        LimitStateResult(
            limit_state=LimitState.FLEXURAL_BUCKLING,
            nominal=Fcr_flexural * Ae,
            citation=cite(area_citation if Ae < Ag else "E3-1"),
            detail={
                "Lc/r": Lc_r_governing,
                "Fe": Fe_flexural,
                "Fcr": Fcr_flexural,
                "Ae": Ae,
                **area_detail,
            },
            note=slenderness_source,
        )
    )

    if Fcr_torsional is not None:
        limit_states.append(
            LimitStateResult(
                limit_state=(
                    LimitState.TORSIONAL_BUCKLING
                    if member.symmetry is Symmetry.DOUBLY
                    else LimitState.FLEXURAL_TORSIONAL_BUCKLING
                ),
                nominal=Fcr_torsional * Ae,
                citation=cite("E4-1"),
                detail={**torsional_detail, "Ae": Ae},
                note=torsional_citation,
            )
        )

    result = StrengthResult.build(
        "Pn",
        "kip",
        limit_states,
        phi=PHI_C,
        omega=OMEGA_C,
        basis=basis,
    )
    if notes:
        # Attach advisories to the governing state so they surface on the report.
        governing = result.governing
        annotated = LimitStateResult(
            limit_state=governing.limit_state,
            nominal=governing.nominal,
            citation=governing.citation,
            detail=governing.detail,
            note="; ".join([governing.note, *notes]) if governing.note else "; ".join(notes),
        )
        result = StrengthResult(
            symbol=result.symbol,
            unit=result.unit,
            phi=result.phi,
            omega=result.omega,
            basis=result.basis,
            limit_states=tuple(
                annotated if ls is governing else ls for ls in result.limit_states
            ),
        )
    return result


def _evaluate_E4(
    member: CompressionMember, E: Ksi, G: Ksi, Ag: Inch2, rx: Inch, ry: Inch
) -> tuple[Ksi | None, dict[str, float], str]:
    """Evaluate the applicable Sect. E4 branch, or return ``None`` if none applies.

    Sect. E4, p. 16.1-36, applies to singly symmetric and unsymmetric members,
    certain doubly symmetric members such as cruciforms and built-up shapes, and
    doubly symmetric members whose torsional effective length exceeds the lateral
    one. For a doubly symmetric rolled shape with ``Lcz <= Lcy`` the User Note to
    Sect. E3 makes clear Sect. E4 will not govern, but it is still evaluated --
    the cost is negligible and reporting it makes the margin explicit.
    """
    section = member.section
    needed = ("Ix", "Iy", "J")
    try:
        props = require_properties(section, *needed)
    except Exception:  # noqa: BLE001 - absence of J/Cw simply means E4 is skipped
        return None, {}, "Sect. E4 not evaluated: Ix, Iy or J unavailable"

    Ix, Iy, J = props["Ix"], props["Iy"], props["J"]
    Cw = float(getattr(section, "Cw", 0.0) or 0.0) if member.include_warping else 0.0
    Lcz = member.Lcz_effective
    if Lcz <= 0.0:
        return None, {}, "Sect. E4 not evaluated: Lcz is zero"

    Fy = member.steel.Fy

    if member.symmetry is Symmetry.DOUBLY:
        Fe = torsional_buckling_stress(E, G, Cw, J, Lcz, Ix, Iy)  # Eq. E4-2
        detail = {"Fe (E4-2)": Fe, "Lcz": Lcz}
        label = "Eq. E4-2, doubly symmetric, twisting about the shear centre"
    else:
        # The User Note on p. 16.1-37 directs xo = 0 for tees and double angles,
        # which is exactly the case in which the warping term is dropped.
        xo = 0.0 if not member.include_warping else member.xo
        ro = polar_radius_of_gyration(xo, member.yo, Ix, Iy, Ag)
        H = flexural_constant(xo, member.yo, ro)
        F_ez = Fez(E, G, Cw, J, Lcz, Ag, ro, include_warping=member.include_warping)  # Eq. E4-7
        F_ex = Fex(member.Lcx / rx, E)  # Eq. E4-5
        F_ey = Fey(member.Lcy / ry, E)  # Eq. E4-6

        if member.symmetry is Symmetry.SINGLY_Y:
            Fe = flexural_torsional_buckling_stress(F_ey, F_ez, H)  # Eq. E4-3
            label = "Eq. E4-3, singly symmetric about y"
        elif member.symmetry is Symmetry.SINGLY_X:
            # User Note to Eq. E4-3, p. 16.1-36: use Fex in place of Fey.
            Fe = flexural_torsional_buckling_stress(F_ex, F_ez, H)  # Eq. E4-3
            label = "Eq. E4-3, singly symmetric about x (Fex substituted for Fey)"
        else:
            Fe = unsymmetric_buckling_stress(F_ex, F_ey, F_ez, xo, member.yo, ro)  # Eq. E4-4
            label = "Eq. E4-4, unsymmetric, lowest root of the cubic"

        detail = {"Fex": F_ex, "Fey": F_ey, "Fez": F_ez, "ro": ro, "H": H, "Fe": Fe}

    Fcr = flexural_buckling_stress(Fy, Fe)  # Eqs. E3-2 / E3-3 applied to the E4 Fe
    detail["Fcr"] = Fcr
    return Fcr, detail, label
