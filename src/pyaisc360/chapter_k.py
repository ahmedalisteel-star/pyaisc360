"""Chapter K -- Additional Requirements for HSS and Box-Section Connections.

Covers Sects. K1 through K5, pp. 16.1-149 to 16.1-164.

Chapter K is **additional to Chapter J, not instead of it** -- the chapter's
opening sentence says so: "The requirements of Chapter J also apply." Every
connection checked here must also satisfy the weld, bolt and element provisions
of Chapter J.

Two things make this chapter unlike the others:

**Its equations live inside tables.** Tables K2.1, K3.1, K3.2, K4.1, K4.2 and
K5.1 carry the limit-state expressions; the section text is mostly pointers to
them. The functions here are grouped by table for that reason.

**Every table has a companion table of limits.** K2.1A, K3.1A, K3.2A, K4.1A,
K4.2A state the ranges of chord slenderness, width ratio, material strength and
ductility within which the table applies. Sect. K1 is explicit about what
falling outside them means: "Connections not complying with the limits of
applicability listed are **not prohibited** and must be designed by rational
analysis." So this module *reports* the violations rather than refusing --
raising would be wrong, because the connection is permitted, just not by these
equations.

The chapter also has an unusually wide factor spread, including a second
``phi = 1.00`` case (Eq. K4-7, chord distortional failure) on top of the four in
Chapters G and J:

=====================================  =====  ======
Limit state                             phi   Omega
=====================================  =====  ======
chord plastification                    0.90   1.67
shear yielding (punching)               0.95   1.58
local yielding due to uneven load       0.95   1.58
chord distortional failure (K4-7)       1.00   1.50
fillet welds to HSS (Sect. K5)          0.75   2.00
PJP groove welds to HSS (Sect. K5)      0.80   1.88
=====================================  =====  ======
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .core.citations import cite
from .core.config import Basis
from .core.enums import LimitState
from .core.exceptions import AISC360Error, GeometryError
from .core.result import InteractionResult, LimitStateResult, StrengthResult
from .core.units import Degrees, Dimensionless, Inch, Inch2, Inch3, Kip, KipIn, Ksi, Ratio

__all__ = [
    "PHI_PLASTIFICATION",
    "OMEGA_PLASTIFICATION",
    "PHI_PUNCHING",
    "OMEGA_PUNCHING",
    "PHI_DISTORTIONAL",
    "OMEGA_DISTORTIONAL",
    "PHI_FILLET_WELD",
    "OMEGA_FILLET_WELD",
    "PHI_PJP_WELD",
    "OMEGA_PJP_WELD",
    "ConnectionType",
    "ApplicabilityCheck",
    "available_stress",
    "effective_width",
    "Qf",
    "utilization_U",
    # K2
    "transverse_plate_to_round_hss",
    "longitudinal_plate_to_round_hss",
    # K3
    "round_punching_shear",
    "round_T_Y_plastification",
    "round_cross_plastification",
    "round_K_plastification",
    "Qg",
    "rectangular_gapped_K_plastification",
    "rectangular_punching_shear",
    "branch_local_yielding",
    "overlapped_K_branch_yielding",
    "overlapped_K_other_branch",
    "beta_eff",
    "beta_eop",
    "Qf_rectangular",
    # K4
    "round_ip_plastification",
    "round_ip_punching",
    "round_op_plastification",
    "round_op_punching",
    "round_moment_interaction",
    "rectangular_chord_distortional",
    "rectangular_moment_interaction",
    # K5
    "weld_axial_strength",
    "weld_ip_moment_strength",
    "weld_op_moment_strength",
    "transverse_plate_weld_length",
    "branch_weld_length",
    "branch_weld_Sip",
    "branch_weld_Sop",
    "gapped_K_weld_length",
    "overlapping_branch_weld_length",
    "overlapped_branch_weld_length",
]

#: Chord plastification -- Tables K2.1, K3.1, K3.2, K4.1.
PHI_PLASTIFICATION: float = 0.90
OMEGA_PLASTIFICATION: float = 1.67
#: Shear yielding (punching) and local yielding due to uneven load distribution.
PHI_PUNCHING: float = 0.95
OMEGA_PUNCHING: float = 1.58
#: Eq. K4-7, chord distortional failure -- the chapter's only phi = 1.00.
PHI_DISTORTIONAL: float = 1.00
OMEGA_DISTORTIONAL: float = 1.50
#: Sect. K5(a) -- fillet welds to rectangular HSS.
PHI_FILLET_WELD: float = 0.75
OMEGA_FILLET_WELD: float = 2.00
#: Sect. K5(b) -- PJP groove welds to rectangular HSS.
PHI_PJP_WELD: float = 0.80
OMEGA_PJP_WELD: float = 1.88


class ConnectionType(str, Enum):
    """Joint configuration. Selects which table row and which limits apply."""

    T = "T-connection"
    Y = "Y-connection"
    CROSS = "cross-connection"
    K_GAPPED = "gapped K-connection"
    K_OVERLAPPED = "overlapped K-connection"


@dataclass(frozen=True, slots=True)
class ApplicabilityCheck:
    """Result of testing a connection against its Table ...A limits.

    Sect. K1, p. 16.1-149: "Connections not complying with the limits of
    applicability listed are **not prohibited** and must be designed by rational
    analysis." A violation therefore reports rather than raises -- the joint is
    permitted, it just cannot be designed by these equations.
    """

    within_limits: bool
    table: str
    violations: tuple[str, ...] = ()

    @property
    def note(self) -> str:
        if self.within_limits:
            return f"within the limits of {self.table}"
        return (
            f"OUTSIDE {self.table}: " + "; ".join(self.violations)
            + " -- Sect. K1 permits the connection but requires rational analysis; "
            "these equations do not apply"
        )


# ===========================================================================
# Sect. K1 -- General provisions and parameters
# ===========================================================================
def available_stress(Fy: Ksi, basis: Basis = Basis.LRFD) -> Ksi:
    """Available stress in the main member, ``Fc``.

    AISC 360-16, Sect. K1.1, p. 16.1-150::

        Fc = Fy for LRFD;  0.60*Fy for ASD

    Note this is **not** the usual ``alpha`` treatment. Elsewhere in the
    Specification an ASD demand is scaled up by 1.6; here the *capacity* is
    scaled down by 0.60, which is 1/1.67 -- the flexural safety factor. ``Fc``
    only ever appears inside Eq. K2-4's utilisation ratio, where a stress
    demand is divided by it, so the effect is the same in the end.
    """
    if Fy <= 0.0:
        raise GeometryError(f"Fy must be positive, got {Fy}")
    return Fy if basis.is_lrfd else 0.60 * Fy


def effective_width(B: Inch, t: Inch, Fy: Ksi, Fyb: Ksi, tb: Inch, Bb: Inch) -> Inch:
    """Effective width of an element transverse to a rectangular HSS.

    AISC 360-16, Eq. K1-1, Sect. K1.2a, p. 16.1-150::

        Be = (10/(B/t))*(Fy*t/(Fyb*tb))*Bb  <=  Bb

    The ``10/(B/t)`` term is the heart of it: a slender chord wall carries the
    transverse force only near the stiff side walls, so only part of the branch
    width is effective. A chord with ``B/t = 10`` is fully effective; one with
    ``B/t = 35`` transfers under a third of it.

    The ``<= Bb`` cap matters -- a stocky chord would otherwise compute an
    effective width wider than the branch itself.
    """
    for name, value in (("B", B), ("t", t), ("Fyb", Fyb), ("tb", tb), ("Bb", Bb)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    Be = (10.0 / (B / t)) * (Fy * t / (Fyb * tb)) * Bb
    return min(Be, Bb)


def utilization_U(Pro: Kip, Mro: KipIn, Fc: Ksi, Ag: Inch2, S: Inch3) -> Ratio:
    """Chord utilisation ratio ``U``.

    AISC 360-16, Eq. K2-4, p. 16.1-151::

        U = Pro/(Fc*Ag) + Mro/(Fc*S)

    ``Pro`` and ``Mro`` are "determined on the side of the joint that has the
    **lower** compression stress" -- not the higher. A chord more heavily loaded
    on one side of the branch is judged by its lighter side, because that is the
    side whose face has capacity left to deform into.
    """
    if Fc <= 0.0 or Ag <= 0.0 or S <= 0.0:
        raise GeometryError(f"Fc, Ag and S must be positive, got {Fc}, {Ag}, {S}")
    return abs(Pro) / (Fc * Ag) + abs(Mro) / (Fc * S)


def Qf(U: Ratio, *, chord_in_tension: bool = False) -> Dimensionless:
    """Chord stress interaction parameter ``Qf``.

    AISC 360-16, Eq. K2-3, p. 16.1-151::

        chord connecting surface in tension:      Qf = 1.0
        chord connecting surface in compression:  Qf = 1.0 - 0.3*U*(1 + U)

    A chord face already carrying compression has less capacity to resist the
    branch pushing into it; one in tension is stiffened by it, so no reduction
    applies. The asymmetry is real and often missed.

    ``Qf`` reaches zero at ``U`` about 1.43, well past any usable chord, and is
    floored at zero rather than allowed negative.
    """
    if U < 0.0:
        raise GeometryError(f"utilisation ratio must be non-negative, got {U}")
    if chord_in_tension:
        return 1.0
    return max(1.0 - 0.3 * U * (1.0 + U), 0.0)


def Qf_rectangular(
    U: Ratio, beta: Ratio, *, chord_in_tension: bool = False, gapped_K: bool = False
) -> Dimensionless:
    """Chord stress interaction parameter for **rectangular** HSS truss connections.

    AISC 360-16, Eqs. K3-14 and K3-15, Table K3.2, p. 16.1-156::

        chord in tension:                        Qf = 1.0
        T-, Y- and cross-connections:            Qf = 1.3 - 0.4*U/beta  <= 1.0   (K3-14)
        gapped K-connections:                    Qf = 1.3 - 0.4*U/beta_eff <= 1.0 (K3-15)

    Structurally different from Eq. K2-3: this one divides ``U`` by the width
    ratio, so a narrow branch on a heavily loaded chord is penalised much harder.
    Using the round-HSS ``Qf`` here is a substitution the tables do not permit.

    Parameters
    ----------
    beta:
        ``Bb/B`` for T-, Y- and cross-connections; ``beta_eff`` from Eq. K3-16
        for gapped K-connections.
    """
    if U < 0.0:
        raise GeometryError(f"utilisation ratio must be non-negative, got {U}")
    if beta <= 0.0:
        raise GeometryError(f"beta must be positive, got {beta}")
    if chord_in_tension:
        return 1.0
    equation = "K3-15" if gapped_K else "K3-14"
    del equation  # both expressions are identical; only the beta differs
    return max(min(1.3 - 0.4 * U / beta, 1.0), 0.0)


def _sin(theta: Degrees) -> float:
    """``sin(theta)`` with the Chapter K angle guard.

    Sect. K1's User Note advises against angles below 30 degrees ("can make
    welding and inspection difficult and should be avoided"), and every table
    divides by ``sin(theta)``, so a zero angle is a division by zero.
    """
    if not 0.0 < theta <= 90.0:
        raise GeometryError(
            f"branch angle must lie in (0, 90] degrees, got {theta}. The Sect. K1 "
            "User Note advises against angles below 30 degrees."
        )
    return math.sin(math.radians(theta))


# ===========================================================================
# Sect. K2 -- Concentrated forces on HSS (Table K2.1)
# ===========================================================================
def transverse_plate_to_round_hss(
    Fy: Ksi, t: Inch, D: Inch, Bb: Inch, theta: Degrees, Qf_value: Dimensionless,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Transverse plate to round HSS, T- and cross-connections.

    AISC 360-16, Table K2.1, Eqs. K2-1a and K2-1b, p. 16.1-151. Limit state:
    HSS local yielding::

        Rn*sin(theta) = Fy*t^2*(5.5/(1 - 0.81*Bb/D))*Qf        (K2-1a)
        Mn = 0.5*Bb*Rn                                          (K2-1b)

    with ``phi = 0.90`` / ``Omega = 1.67``.

    The ``1 - 0.81*Bb/D`` denominator blows up as the plate approaches the tube
    diameter -- at ``Bb/D = 1.0`` it is 0.19, giving a factor of 29. Table K2.1A
    caps the width ratio at 1.0 for exactly that reason, and the equation is
    meaningless above it.

    Returns the **axial** strength ``Rn``; the in-plane bending strength is
    ``0.5*Bb*Rn`` from Eq. K2-1b, reported in the detail.
    """
    for name, value in (("Fy", Fy), ("t", t), ("D", D), ("Bb", Bb)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    beta = Bb / D
    denominator = 1.0 - 0.81 * beta
    if denominator <= 0.0:
        raise GeometryError(
            f"Bb/D = {beta:.3f} drives 1 - 0.81*Bb/D to {denominator:.4f}. Eq. K2-1a "
            "is singular at Bb/D = 1.235; Table K2.1A caps the ratio at 1.0."
        )

    Rn = Fy * t**2 * (5.5 / denominator) * Qf_value / _sin(theta)  # Eq. K2-1a
    state = LimitStateResult(
        LimitState.WEB_LOCAL_YIELDING, Rn, cite("K2-1a"),
        detail={"Bb/D": beta, "Qf": Qf_value, "Mn (Eq. K2-1b)": 0.5 * Bb * Rn},
        note="HSS local yielding; Mn = 0.5*Bb*Rn for in-plane bending (Eq. K2-1b)",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Rn", "kip", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def longitudinal_plate_to_round_hss(
    Fy: Ksi, t: Inch, D: Inch, lb: Inch, theta: Degrees, Qf_value: Dimensionless,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Longitudinal plate to round HSS, T-, Y- and cross-connections.

    AISC 360-16, Table K2.1, Eqs. K2-2a and K2-2b, p. 16.1-151. Limit state:
    HSS plastification::

        Rn*sin(theta) = 5.5*Fy*t^2*(1 + 0.25*lb/D)*Qf           (K2-2a)
        Mn = 0.8*lb*Rn                                           (K2-2b)

    with ``phi = 0.90`` / ``Omega = 1.67``.

    The opposite sign to the transverse case: bearing length *helps* here
    (``1 + 0.25*lb/D``) where branch width hurt there (``1/(1 - 0.81*Bb/D)``).
    A longitudinal plate loads the tube along its stiff meridian.

    Note the moment coefficient differs too -- ``0.8*lb`` against Eq. K2-1b's
    ``0.5*Bb``.

    Notes
    -----
    Eq. K2-2b's label is misprinted as "(K2-2a)" in the machine-readable text
    dump of the Specification; ``tools/scan_spec.py::EXTRACTION_CORRECTIONS``
    records the evidence and restores it.
    """
    for name, value in (("Fy", Fy), ("t", t), ("D", D), ("lb", lb)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    Rn = 5.5 * Fy * t**2 * (1.0 + 0.25 * lb / D) * Qf_value / _sin(theta)  # Eq. K2-2a
    state = LimitStateResult(
        LimitState.WEB_LOCAL_YIELDING, Rn, cite("K2-2a"),
        detail={"lb/D": lb / D, "Qf": Qf_value, "Mn (Eq. K2-2b)": 0.8 * lb * Rn},
        note="HSS plastification; Mn = 0.8*lb*Rn for in-plane bending (Eq. K2-2b)",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Rn", "kip", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


# ===========================================================================
# Sect. K3 -- HSS-to-HSS truss connections (Tables K3.1 and K3.2)
# ===========================================================================
def round_punching_shear(
    Fy: Ksi, t: Inch, Db: Inch, theta: Degrees, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Shear yielding (punching) of a round HSS chord.

    AISC 360-16, Table K3.1, Eq. K3-1, p. 16.1-154::

        Pn*sin(theta) = 0.6*Fy*t*pi*Db*((1 + sin(theta))/(2*sin^2(theta)))

    with ``phi = 0.95`` / ``Omega = 1.58``.

    Checked only "when ``Db < D - 2t``" -- a branch as wide as the chord bore
    cannot punch through it, it simply bears on the far wall.
    """
    for name, value in (("Fy", Fy), ("t", t), ("Db", Db)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    s = _sin(theta)
    Pn = 0.6 * Fy * t * math.pi * Db * ((1.0 + s) / (2.0 * s**2)) / s  # Eq. K3-1
    state = LimitStateResult(
        LimitState.SHEAR_YIELDING, Pn, cite("K3-1"),
        detail={"theta": theta},
        note="punching shear; checked only when Db < D - 2t",
        phi=PHI_PUNCHING, omega=OMEGA_PUNCHING,
    )
    return StrengthResult.build(
        "Pn", "kip", [state], phi=PHI_PUNCHING, omega=OMEGA_PUNCHING, basis=basis
    )


def round_T_Y_plastification(
    Fy: Ksi, t: Inch, beta: Ratio, gamma: Ratio, theta: Degrees, Qf_value: Dimensionless,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Chord plastification, round HSS T- and Y-connections.

    AISC 360-16, Table K3.1, Eq. K3-2, p. 16.1-154::

        Pn*sin(theta) = Fy*t^2*(3.1 + 15.6*beta^2)*gamma^0.2*Qf

    with ``phi = 0.90`` / ``Omega = 1.67``, where ``beta = Db/D`` and
    ``gamma = D/(2t)``.

    Note ``gamma`` enters as ``gamma^0.2`` -- a *positive* power, so a more
    slender chord is computed **stronger** by this term. That is not a mistake:
    the ``Fy*t^2`` prefactor already falls as ``t^2``, and the fifth root only
    partially offsets it. Net of both, a thinner chord is much weaker.
    """
    if Fy <= 0.0 or t <= 0.0:
        raise GeometryError(f"Fy and t must be positive, got {Fy}, {t}")
    if beta <= 0.0 or gamma <= 0.0:
        raise GeometryError(f"beta and gamma must be positive, got {beta}, {gamma}")

    Pn = Fy * t**2 * (3.1 + 15.6 * beta**2) * gamma**0.2 * Qf_value / _sin(theta)
    state = LimitStateResult(
        LimitState.YIELDING, Pn, cite("K3-2"),
        detail={"beta": beta, "gamma": gamma, "Qf": Qf_value},
        note="chord plastification, T- and Y-connections",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Pn", "kip", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def round_cross_plastification(
    Fy: Ksi, t: Inch, beta: Ratio, theta: Degrees, Qf_value: Dimensionless,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Chord plastification, round HSS cross-connections.

    AISC 360-16, Table K3.1, Eq. K3-3, p. 16.1-154::

        Pn*sin(theta) = Fy*t^2*(5.7/(1 - 0.81*beta))*Qf

    with ``phi = 0.90`` / ``Omega = 1.67``.

    Same ``1 - 0.81*beta`` denominator as Eq. K2-1a, and the same singularity at
    ``beta = 1.235``. A cross-connection has branches on both faces, so the
    chord cannot develop the ring action a T-connection relies on -- which is
    why this expression has no ``gamma`` term at all.
    """
    if Fy <= 0.0 or t <= 0.0:
        raise GeometryError(f"Fy and t must be positive, got {Fy}, {t}")
    denominator = 1.0 - 0.81 * beta
    if denominator <= 0.0:
        raise GeometryError(
            f"beta = {beta:.3f} makes 1 - 0.81*beta non-positive; Eq. K3-3 is "
            "singular at beta = 1.235 and Table K3.1A caps beta at 1.0"
        )

    Pn = Fy * t**2 * (5.7 / denominator) * Qf_value / _sin(theta)
    state = LimitStateResult(
        LimitState.YIELDING, Pn, cite("K3-3"),
        detail={"beta": beta, "Qf": Qf_value},
        note="chord plastification, cross-connections",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Pn", "kip", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def Qg(gamma: Ratio, g: Inch, t: Inch) -> Dimensionless:
    """Gap parameter ``Qg`` for round HSS K-connections.

    AISC 360-16, Table K3.1, Eq. K3-6, p. 16.1-154::

        Qg = gamma^0.2 * [1 + 0.024*gamma^1.2/(exp(0.5*g/t - 1.33) + 1)]

    The exponential makes this the most numerically awkward expression in the
    chapter. Two behaviours matter:

    * a **small gap** (``g/t`` small) drives the exponent negative,
      ``exp(...) -> 0``, and the bracket towards ``1 + 0.024*gamma^1.2`` -- its
      maximum. Closely spaced branches share chord material and the joint is
      stronger.
    * a **large gap** drives ``exp(...) -> inf`` and the bracket to 1, leaving
      ``Qg = gamma^0.2``.

    ``exp`` overflows for ``0.5*g/t`` above about 710, so the large-gap limit is
    taken directly rather than evaluated -- an overflow there would produce
    ``inf`` and then ``nan``, not a large number.
    """
    if gamma <= 0.0 or t <= 0.0:
        raise GeometryError(f"gamma and t must be positive, got {gamma}, {t}")
    if g < 0.0:
        raise GeometryError(f"gap must be non-negative, got {g}")

    exponent = 0.5 * g / t - 1.33
    # math.pow rather than **: mypy widens ** to Any because a negative base with
    # a fractional exponent is complex. gamma > 0 is guarded above.
    base = math.pow(gamma, 0.2)
    if exponent > 700.0:
        # exp() would overflow to inf and then produce nan; the bracket has
        # already converged to 1 by this point.
        return base
    return base * (1.0 + 0.024 * math.pow(gamma, 1.2) / (math.exp(exponent) + 1.0))


def round_K_plastification(
    Fy: Ksi, t: Inch, Db_comp: Inch, D: Inch, theta: Degrees,
    Qg_value: Dimensionless, Qf_value: Dimensionless, *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Chord plastification, round HSS K-connections with gap or overlap.

    AISC 360-16, Table K3.1, Eqs. K3-4 and K3-5, p. 16.1-154::

        (Pn*sin(theta))_compression branch
            = Fy*t^2*(2.0 + 11.33*Db,comp/D)*Qg*Qf                (K3-4)
        (Pn*sin(theta))_tension branch
            = (Pn*sin(theta))_compression branch                   (K3-5)

    with ``phi = 0.90`` / ``Omega = 1.67``.

    Eq. K3-5 is the one people drop: the **tension branch takes the compression
    branch's value**, computed from the *compression* branch's diameter. It is
    not evaluated independently. A K-joint fails by the chord face deforming
    under the compression branch, and the tension branch can carry no more than
    that mechanism allows.
    """
    if Fy <= 0.0 or t <= 0.0 or D <= 0.0 or Db_comp <= 0.0:
        raise GeometryError("Fy, t, D and Db,comp must be positive")

    Pn = Fy * t**2 * (2.0 + 11.33 * Db_comp / D) * Qg_value * Qf_value / _sin(theta)
    state = LimitStateResult(
        LimitState.YIELDING, Pn, cite("K3-4"),
        detail={"Db,comp/D": Db_comp / D, "Qg": Qg_value, "Qf": Qf_value},
        note=(
            "chord plastification; Eq. K3-5 gives the TENSION branch this same "
            "value, computed from the COMPRESSION branch's diameter"
        ),
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Pn", "kip", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def beta_eff(
    Bb_comp: Inch, Hb_comp: Inch, Bb_tens: Inch, Hb_tens: Inch, B: Inch
) -> Ratio:
    """Effective width ratio for a gapped rectangular K-connection.

    AISC 360-16, Eq. K3-16, Table K3.2, p. 16.1-156::

        beta_eff = [(Bb + Hb)_compression + (Bb + Hb)_tension] / (4*B)

    Both branches contribute, so a K-joint's effective width ratio depends on
    the *pair*, not on either branch alone.
    """
    if B <= 0.0:
        raise GeometryError(f"B must be positive, got {B}")
    for name, value in (
        ("Bb,comp", Bb_comp), ("Hb,comp", Hb_comp), ("Bb,tens", Bb_tens), ("Hb,tens", Hb_tens)
    ):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    return ((Bb_comp + Hb_comp) + (Bb_tens + Hb_tens)) / (4.0 * B)


def beta_eop(beta: Ratio, gamma: Ratio) -> Ratio:
    """Effective outside punching parameter.

    AISC 360-16, Eq. K3-17, Table K3.2, p. 16.1-156::

        beta_eop = 5*beta/gamma  <=  beta
    """
    if gamma <= 0.0:
        raise GeometryError(f"gamma must be positive, got {gamma}")
    return min(5.0 * beta / gamma, beta)


def rectangular_gapped_K_plastification(
    Fy: Ksi, t: Inch, beta_eff_value: Ratio, gamma: Ratio, theta: Degrees,
    Qf_value: Dimensionless, *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Chord wall plastification, gapped rectangular HSS K-connections.

    AISC 360-16, Table K3.2, Eq. K3-7, p. 16.1-155::

        Pn*sin(theta) = Fy*t^2*(9.8*beta_eff*gamma^0.5)*Qf

    with ``phi = 0.90`` / ``Omega = 1.67``. Applies "for all beta" -- there is
    no width-ratio branching here, unlike the round-HSS tables.
    """
    if Fy <= 0.0 or t <= 0.0 or gamma <= 0.0:
        raise GeometryError("Fy, t and gamma must be positive")

    Pn = Fy * t**2 * (9.8 * beta_eff_value * math.sqrt(gamma)) * Qf_value / _sin(theta)
    state = LimitStateResult(
        LimitState.YIELDING, Pn, cite("K3-7"),
        detail={"beta_eff": beta_eff_value, "gamma": gamma, "Qf": Qf_value},
        note="chord wall plastification, for all beta",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Pn", "kip", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def rectangular_punching_shear(
    Fy: Ksi, t: Inch, B: Inch, eta: Ratio, beta: Ratio, beta_eop_value: Ratio,
    theta: Degrees, *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Shear yielding (punching), rectangular HSS truss connections.

    AISC 360-16, Table K3.2, Eq. K3-8, p. 16.1-155::

        Pn*sin(theta) = 0.6*Fy*t*B*(2*eta + beta + beta_eop)

    with ``phi = 0.95`` / ``Omega = 1.58``.

    Checked only "when ``Bb < B - 2t``", and Table K3.2 adds that "this limit
    state need not be checked for square branches" -- a square branch punches on
    all four faces at once and the chord side walls prevent it.
    """
    if Fy <= 0.0 or t <= 0.0 or B <= 0.0:
        raise GeometryError("Fy, t and B must be positive")

    Pn = 0.6 * Fy * t * B * (2.0 * eta + beta + beta_eop_value) / _sin(theta)
    state = LimitStateResult(
        LimitState.SHEAR_YIELDING, Pn, cite("K3-8"),
        detail={"eta": eta, "beta": beta, "beta_eop": beta_eop_value},
        note="punching shear; not required for square branches, or when Bb >= B - 2t",
        phi=PHI_PUNCHING, omega=OMEGA_PUNCHING,
    )
    return StrengthResult.build(
        "Pn", "kip", [state], phi=PHI_PUNCHING, omega=OMEGA_PUNCHING, basis=basis
    )


def branch_local_yielding(
    Fyb: Ksi, tb: Inch, Hb: Inch, Bb: Inch, Be: Inch, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Local yielding of the branch due to uneven load distribution.

    AISC 360-16, Table K3.2, Eq. K3-9, p. 16.1-155::

        Pn = Fyb*tb*(2*Hb + Bb + Be - 4*tb)

    with ``phi = 0.95`` / ``Omega = 1.58``.

    Table K3.2 exempts square branches and any connection with ``B/t >= 15``.
    The ``- 4*tb`` removes the corners, which are counted twice by the
    perimeter sum.

    Note ``Be < Bb`` always (Eq. K1-1), so the effective perimeter is always
    less than the true one -- the branch face bearing on a flexible chord wall
    only engages near the stiff side walls.
    """
    for name, value in (("Fyb", Fyb), ("tb", tb), ("Hb", Hb), ("Bb", Bb), ("Be", Be)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    perimeter = 2.0 * Hb + Bb + Be - 4.0 * tb
    if perimeter <= 0.0:
        raise GeometryError(
            f"effective perimeter came out non-positive ({perimeter:.4g} in.); "
            "check that Hb, Bb and Be are branch dimensions, not chord ones"
        )
    state = LimitStateResult(
        LimitState.YIELDING, Fyb * tb * perimeter, cite("K3-9"),
        detail={"Be": Be, "effective perimeter": perimeter},
        note="uneven load distribution; not required for square branches or B/t >= 15",
        phi=PHI_PUNCHING, omega=OMEGA_PUNCHING,
    )
    return StrengthResult.build(
        "Pn", "kip", [state], phi=PHI_PUNCHING, omega=OMEGA_PUNCHING, basis=basis
    )


def overlapped_K_branch_yielding(
    Fybi: Ksi, tbi: Inch, Hbi: Inch, Bbi: Inch, Bei: Inch, Bej: Inch, Ov: Ratio,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Local yielding of the **overlapping** branch of a K-connection.

    AISC 360-16, Table K3.2, Eqs. K3-10, K3-11 and K3-12, p. 16.1-156::

        25% <= Ov < 50%:   Pn,i = Fybi*tbi*[(Ov/50)*(2*Hbi - 4*tbi)
                                            + Bei + Bej]           (K3-10)
        50% <= Ov < 80%:   Pn,i = Fybi*tbi*(2*Hbi - 4*tbi + Bei + Bej)  (K3-11)
        80% <= Ov <= 100%: Pn,i = Fybi*tbi*(2*Hbi - 4*tbi + Bbi + Bej)  (K3-12)

    with ``phi = 0.95`` / ``Omega = 1.58``.

    The three branches trace how the load path changes as overlap grows. Below
    50% the web contribution is *pro-rated* by ``Ov/50``; from 50% it counts in
    full; and from 80% the overlapping branch's own width switches from the
    effective ``Bei`` to the **full** ``Bbi``, because at that overlap it bears
    on the other branch rather than on the flexible chord face.

    ``Ov`` is a percentage (25 to 100), not a fraction -- Table K3.2A limits it
    to that range.

    Raises
    ------
    AISC360Error
        Below 25% overlap, where Table K3.2A's limits do not reach and no
        equation is given.
    """
    if not 25.0 <= Ov <= 100.0:
        raise AISC360Error(
            f"overlap Ov = {Ov} is outside the 25%-100% range of Table K3.2A. "
            "Ov is a percentage, not a fraction."
        )
    for name, value in (("Fybi", Fybi), ("tbi", tbi), ("Hbi", Hbi), ("Bbi", Bbi)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    web = 2.0 * Hbi - 4.0 * tbi
    if Ov < 50.0:
        Pn = Fybi * tbi * ((Ov / 50.0) * web + Bei + Bej)  # Eq. K3-10
        citation, note = cite("K3-10"), f"Ov = {Ov:g}%: web contribution pro-rated by Ov/50"
    elif Ov < 80.0:
        Pn = Fybi * tbi * (web + Bei + Bej)  # Eq. K3-11
        citation, note = cite("K3-11"), f"Ov = {Ov:g}%: full web contribution"
    else:
        Pn = Fybi * tbi * (web + Bbi + Bej)  # Eq. K3-12
        citation, note = cite("K3-12"), (
            f"Ov = {Ov:g}%: the overlapping branch's FULL width Bbi replaces Bei -- "
            "it bears on the other branch, not the chord face"
        )

    state = LimitStateResult(
        LimitState.YIELDING, max(Pn, 0.0), citation,
        detail={"Ov": Ov, "Bei": Bei, "Bej": Bej},
        note=note, phi=PHI_PUNCHING, omega=OMEGA_PUNCHING,
    )
    return StrengthResult.build(
        "Pn", "kip", [state], phi=PHI_PUNCHING, omega=OMEGA_PUNCHING, basis=basis
    )


def overlapped_K_other_branch(
    Pn_i: Kip, Fybi: Ksi, Abi: Inch2, Fybj: Ksi, Abj: Inch2
) -> Kip:
    """Strength of the **overlapped** branch, from the overlapping branch's.

    AISC 360-16, Eq. K3-13, Table K3.2, p. 16.1-156::

        Pn,j = Pn,i*(Fybj*Abj)/(Fybi*Abi)

    The overlapped branch is not checked independently -- its strength is scaled
    from the overlapping branch's by the ratio of squash loads. Subscript ``i``
    is the **overlapping** branch and ``j`` the **overlapped** one; swapping
    them inverts the ratio.
    """
    for name, value in (("Fybi", Fybi), ("Abi", Abi), ("Fybj", Fybj), ("Abj", Abj)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    return Pn_i * (Fybj * Abj) / (Fybi * Abi)


# ===========================================================================
# Sect. K4 -- HSS-to-HSS moment connections (Tables K4.1 and K4.2)
# ===========================================================================
def round_ip_plastification(
    Fy: Ksi, t: Inch, Db: Inch, beta: Ratio, gamma: Ratio, theta: Degrees,
    Qf_value: Dimensionless, *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Chord plastification under in-plane bending, round HSS.

    AISC 360-16, Table K4.1, Eq. K4-1, p. 16.1-159::

        Mn-ip*sin(theta) = 5.39*Fy*t^2*gamma^0.5*beta*Db*Qf

    with ``phi = 0.90`` / ``Omega = 1.67``.

    Note ``gamma^0.5`` here against Eq. K3-2's ``gamma^0.2`` -- bending mobilises
    chord slenderness more strongly than axial load does.
    """
    if Fy <= 0.0 or t <= 0.0 or Db <= 0.0 or gamma <= 0.0:
        raise GeometryError("Fy, t, Db and gamma must be positive")
    Mn = 5.39 * Fy * t**2 * math.sqrt(gamma) * beta * Db * Qf_value / _sin(theta)
    state = LimitStateResult(
        LimitState.YIELDING, Mn, cite("K4-1"),
        detail={"beta": beta, "gamma": gamma, "Qf": Qf_value},
        note="chord plastification, in-plane bending",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Mn-ip", "kip-in.", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def round_ip_punching(
    Fy: Ksi, t: Inch, Db: Inch, theta: Degrees, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Punching shear under in-plane bending, round HSS.

    AISC 360-16, Table K4.1, Eq. K4-2, p. 16.1-159::

        Mn-ip = 0.6*Fy*t*Db^2*((1 + 3*sin(theta))/(4*sin^2(theta)))

    with ``phi = 0.95`` / ``Omega = 1.58``. Checked only when ``Db < D - 2t``.
    """
    if Fy <= 0.0 or t <= 0.0 or Db <= 0.0:
        raise GeometryError("Fy, t and Db must be positive")
    s = _sin(theta)
    Mn = 0.6 * Fy * t * Db**2 * ((1.0 + 3.0 * s) / (4.0 * s**2))
    state = LimitStateResult(
        LimitState.SHEAR_YIELDING, Mn, cite("K4-2"),
        detail={"theta": theta}, note="punching shear, in-plane bending",
        phi=PHI_PUNCHING, omega=OMEGA_PUNCHING,
    )
    return StrengthResult.build(
        "Mn-ip", "kip-in.", [state], phi=PHI_PUNCHING, omega=OMEGA_PUNCHING, basis=basis
    )


def round_op_plastification(
    Fy: Ksi, t: Inch, Db: Inch, beta: Ratio, theta: Degrees, Qf_value: Dimensionless,
    *, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Chord plastification under out-of-plane bending, round HSS.

    AISC 360-16, Table K4.1, Eq. K4-3, p. 16.1-159::

        Mn-op*sin(theta) = Fy*t^2*Db*(3.0/(1 - 0.81*beta))*Qf

    with ``phi = 0.90`` / ``Omega = 1.67``. Same ``1 - 0.81*beta`` singularity
    as Eqs. K2-1a and K3-3.
    """
    if Fy <= 0.0 or t <= 0.0 or Db <= 0.0:
        raise GeometryError("Fy, t and Db must be positive")
    denominator = 1.0 - 0.81 * beta
    if denominator <= 0.0:
        raise GeometryError(
            f"beta = {beta:.3f} makes 1 - 0.81*beta non-positive; Eq. K4-3 is "
            "singular at beta = 1.235"
        )
    Mn = Fy * t**2 * Db * (3.0 / denominator) * Qf_value / _sin(theta)
    state = LimitStateResult(
        LimitState.YIELDING, Mn, cite("K4-3"),
        detail={"beta": beta, "Qf": Qf_value},
        note="chord plastification, out-of-plane bending",
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION,
    )
    return StrengthResult.build(
        "Mn-op", "kip-in.", [state],
        phi=PHI_PLASTIFICATION, omega=OMEGA_PLASTIFICATION, basis=basis,
    )


def round_op_punching(
    Fy: Ksi, t: Inch, Db: Inch, theta: Degrees, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Punching shear under out-of-plane bending, round HSS.

    AISC 360-16, Table K4.1, Eq. K4-4, p. 16.1-159::

        Mn-op = 0.6*Fy*t*Db^2*((3 + sin(theta))/(4*sin^2(theta)))

    with ``phi = 0.95`` / ``Omega = 1.58``.

    Compare Eq. K4-2: the numerator is ``3 + sin(theta)`` here against
    ``1 + 3*sin(theta)`` there. The two swap at ``theta = 90``, where both give
    4, and diverge as the angle drops -- out-of-plane bending is the more severe
    case for an inclined branch.
    """
    if Fy <= 0.0 or t <= 0.0 or Db <= 0.0:
        raise GeometryError("Fy, t and Db must be positive")
    s = _sin(theta)
    Mn = 0.6 * Fy * t * Db**2 * ((3.0 + s) / (4.0 * s**2))
    state = LimitStateResult(
        LimitState.SHEAR_YIELDING, Mn, cite("K4-4"),
        detail={"theta": theta}, note="punching shear, out-of-plane bending",
        phi=PHI_PUNCHING, omega=OMEGA_PUNCHING,
    )
    return StrengthResult.build(
        "Mn-op", "kip-in.", [state], phi=PHI_PUNCHING, omega=OMEGA_PUNCHING, basis=basis
    )


def round_moment_interaction(
    Pr: Kip, Pc: Kip, Mr_ip: KipIn, Mc_ip: KipIn, Mr_op: KipIn, Mc_op: KipIn,
    basis: Basis = Basis.LRFD,
) -> InteractionResult:
    """Combined axial and biaxial bending, **round** HSS-to-HSS connections.

    AISC 360-16, Table K4.1, Eqs. K4-5 (LRFD) and K4-6 (ASD), p. 16.1-159::

        Pr/Pc + (Mr-ip/Mc-ip)^2 + Mr-op/Mc-op  <=  1.0

    The **in-plane term is squared and the out-of-plane term is not**. That
    asymmetry is the whole point of the equation and is easy to lose: squaring
    both, or neither, changes the answer materially. In-plane bending shares the
    chord's ring-bending mechanism with axial load, so it interacts weakly;
    out-of-plane bending twists the chord and adds linearly.

    Contrast :func:`rectangular_moment_interaction`, where **neither** term is
    squared.
    """
    for name, value in (("Pc", Pc), ("Mc-ip", Mc_ip), ("Mc-op", Mc_op)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    axial = abs(Pr) / Pc
    ip = abs(Mr_ip) / Mc_ip
    op = abs(Mr_op) / Mc_op
    return InteractionResult(
        ratio=axial + ip**2 + op,
        citation=cite("K4-5" if basis.is_lrfd else "K4-6"),
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE,
        terms={"Pr/Pc": axial, "(Mr-ip/Mc-ip)^2": ip**2, "Mr-op/Mc-op": op},
        note="round HSS: the IN-PLANE term is squared, the out-of-plane one is not",
    )


def rectangular_chord_distortional(
    Fy: Ksi, t: Inch, Hb: Inch, B: Inch, H: Inch, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Chord distortional failure under out-of-plane bending, rectangular HSS.

    AISC 360-16, Table K4.2, Eq. K4-7, p. 16.1-161::

        Mn = 2*Fy*t*[Hb*t + sqrt(B*H*t*(B + H))]

    with ``phi = 1.00`` / ``Omega = 1.50`` -- the chapter's only ``phi = 1.00``,
    and the fifth in the whole Specification alongside Sects. G2.1(a), J3.8,
    J4.2(a) and J10.2.

    Applies to "T-connections and **unbalanced** cross-connections". A balanced
    cross-connection has equal and opposite branch moments, so the chord is not
    driven into a rhombus and this mode does not arise.
    """
    for name, value in (("Fy", Fy), ("t", t), ("Hb", Hb), ("B", B), ("H", H)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    Mn = 2.0 * Fy * t * (Hb * t + math.sqrt(B * H * t * (B + H)))
    state = LimitStateResult(
        LimitState.YIELDING, Mn, cite("K4-7"),
        detail={"Hb*t": Hb * t, "sqrt(B*H*t*(B+H))": math.sqrt(B * H * t * (B + H))},
        note=(
            "chord distortional failure; T-connections and UNBALANCED "
            "cross-connections only"
        ),
        phi=PHI_DISTORTIONAL, omega=OMEGA_DISTORTIONAL,
    )
    return StrengthResult.build(
        "Mn", "kip-in.", [state],
        phi=PHI_DISTORTIONAL, omega=OMEGA_DISTORTIONAL, basis=basis,
    )


def rectangular_moment_interaction(
    Pr: Kip, Pc: Kip, Mr_ip: KipIn, Mc_ip: KipIn, Mr_op: KipIn, Mc_op: KipIn,
    basis: Basis = Basis.LRFD,
) -> InteractionResult:
    """Combined axial and biaxial bending, **rectangular** HSS-to-HSS connections.

    AISC 360-16, Table K4.2, Eqs. K4-8 (LRFD) and K4-9 (ASD), p. 16.1-161::

        Pr/Pc + Mr-ip/Mc-ip + Mr-op/Mc-op  <=  1.0

    Fully linear -- **no squared term**, unlike Eqs. K4-5/K4-6 for round HSS. A
    rectangular chord face has no ring action to share between axial load and
    in-plane bending, so the three effects simply add.

    Using the round-HSS form here is unconservative wherever in-plane bending is
    significant, because squaring a ratio below 1.0 reduces it.
    """
    for name, value in (("Pc", Pc), ("Mc-ip", Mc_ip), ("Mc-op", Mc_op)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    axial = abs(Pr) / Pc
    ip = abs(Mr_ip) / Mc_ip
    op = abs(Mr_op) / Mc_op
    return InteractionResult(
        ratio=axial + ip + op,
        citation=cite("K4-8" if basis.is_lrfd else "K4-9"),
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE,
        terms={"Pr/Pc": axial, "Mr-ip/Mc-ip": ip, "Mr-op/Mc-op": op},
        note="rectangular HSS: fully linear, NO squared term (contrast Eq. K4-5)",
    )


# ===========================================================================
# Sect. K5 -- Welds of plates and branches to rectangular HSS
# ===========================================================================
def weld_axial_strength(
    Fnw: Ksi, tw: Inch, le: Inch, *, pjp: bool = False, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Weld strength for an axially loaded branch or plate on rectangular HSS.

    AISC 360-16, Eq. K5-1, Sect. K5, p. 16.1-158::

        Rn or Pn = Fnw*tw*le

    with ``phi = 0.75`` / ``Omega = 2.00`` for fillet welds and ``0.80`` /
    ``1.88`` for PJP groove welds.

    ``Fnw`` is the Chapter J nominal weld stress **with no directional strength
    increase** -- Sect. K5 says so explicitly. The Eq. J2-5 ``(1 + 0.5*sin^1.5
    theta)`` bonus is not available here, because the load along the weld is
    already non-uniform and the effective length ``le`` accounts for it instead.
    Applying both double-counts.

    ``le`` is the *effective* weld length from Table K5.1, not the physical one.
    """
    if Fnw <= 0.0 or tw <= 0.0 or le <= 0.0:
        raise GeometryError(f"Fnw, tw and le must be positive, got {Fnw}, {tw}, {le}")

    phi = PHI_PJP_WELD if pjp else PHI_FILLET_WELD
    omega = OMEGA_PJP_WELD if pjp else OMEGA_FILLET_WELD
    state = LimitStateResult(
        LimitState.WELD_RUPTURE, Fnw * tw * le, cite("K5-1"),
        detail={"Fnw": Fnw, "tw": tw, "le": le},
        note=(
            f"{'PJP groove' if pjp else 'fillet'} weld; Fnw carries NO directional "
            "strength increase (Sect. K5)"
        ),
        phi=phi, omega=omega,
    )
    return StrengthResult.build("Rn", "kip", [state], phi=phi, omega=omega, basis=basis)


def weld_ip_moment_strength(
    Fnw: Ksi, Sip: Inch3, *, pjp: bool = False, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Weld strength under in-plane bending. Eq. K5-2, Sect. K5, p. 16.1-158.

    ``Mn-ip = Fnw*Sip``, with ``Sip`` from Table K5.1.
    """
    if Fnw <= 0.0 or Sip <= 0.0:
        raise GeometryError(f"Fnw and Sip must be positive, got {Fnw}, {Sip}")
    phi = PHI_PJP_WELD if pjp else PHI_FILLET_WELD
    omega = OMEGA_PJP_WELD if pjp else OMEGA_FILLET_WELD
    state = LimitStateResult(
        LimitState.WELD_RUPTURE, Fnw * Sip, cite("K5-2"),
        detail={"Sip": Sip}, note="in-plane bending; Sect. K5 requires interaction "
        "with the axial and out-of-plane effects to be considered",
        phi=phi, omega=omega,
    )
    return StrengthResult.build("Mn-ip", "kip-in.", [state], phi=phi, omega=omega, basis=basis)


def weld_op_moment_strength(
    Fnw: Ksi, Sop: Inch3, *, pjp: bool = False, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Weld strength under out-of-plane bending. Eq. K5-3, Sect. K5, p. 16.1-158.

    ``Mn-op = Fnw*Sop``, with ``Sop`` from Table K5.1.
    """
    if Fnw <= 0.0 or Sop <= 0.0:
        raise GeometryError(f"Fnw and Sop must be positive, got {Fnw}, {Sop}")
    phi = PHI_PJP_WELD if pjp else PHI_FILLET_WELD
    omega = OMEGA_PJP_WELD if pjp else OMEGA_FILLET_WELD
    state = LimitStateResult(
        LimitState.WELD_RUPTURE, Fnw * Sop, cite("K5-3"),
        detail={"Sop": Sop}, note="out-of-plane bending",
        phi=phi, omega=omega,
    )
    return StrengthResult.build("Mn-op", "kip-in.", [state], phi=phi, omega=omega, basis=basis)


def transverse_plate_weld_length(Be: Inch) -> Inch:
    """Effective weld length for a transverse plate T- or cross-connection.

    AISC 360-16, Eq. K5-4, Table K5.1, p. 16.1-163::

        le = 2*Be

    "where le = total effective weld length for welds on **both sides** of the
    transverse plate" -- the factor of two is the two sides, not two welds per
    side.
    """
    if Be <= 0.0:
        raise GeometryError(f"Be must be positive, got {Be}")
    return 2.0 * Be


def branch_weld_length(Hb: Inch, Be: Inch, theta: Degrees) -> Inch:
    """Effective weld length for a T-, Y- or cross-connection branch.

    AISC 360-16, Eq. K5-5, Table K5.1, p. 16.1-163::

        le = 2*Hb/sin(theta) + 2*Be

    The webs are divided by ``sin(theta)`` because an inclined branch presents a
    longer weld along the chord face; the flanges use the *effective* width and
    are not.
    """
    if Hb <= 0.0 or Be <= 0.0:
        raise GeometryError(f"Hb and Be must be positive, got {Hb}, {Be}")
    return 2.0 * Hb / _sin(theta) + 2.0 * Be


def branch_weld_Sip(tw: Inch, Hb: Inch, Be: Inch, theta: Degrees) -> Inch3:
    """Effective elastic section modulus of the welds for in-plane bending.

    AISC 360-16, Eq. K5-6, Table K5.1, p. 16.1-163::

        Sip = (tw/3)*(Hb/sin(theta))^2 + tw*Be*(Hb/sin(theta))
    """
    if tw <= 0.0 or Hb <= 0.0 or Be <= 0.0:
        raise GeometryError(f"tw, Hb and Be must be positive, got {tw}, {Hb}, {Be}")
    h = Hb / _sin(theta)
    return (tw / 3.0) * h**2 + tw * Be * h


def branch_weld_Sop(tw: Inch, Hb: Inch, Bb: Inch, Be: Inch, theta: Degrees) -> Inch3:
    """Effective elastic section modulus of the welds for out-of-plane bending.

    AISC 360-16, Eq. K5-7, Table K5.1, p. 16.1-163::

        Sop = tw*(Hb/sin(theta))*Bb + (tw/3)*(Bb^2) - (tw/3)*((Bb - Be)^3/Bb)

    Table K5.1 adds a proviso: "When ``beta > 0.85`` or ``theta > 50`` degrees,
    ``Be/2`` shall not exceed ``Bb/4``" -- see :func:`branch_weld_Sop` callers,
    which must clamp ``Be`` accordingly before calling.
    """
    if min(tw, Hb, Bb, Be) <= 0.0:
        raise GeometryError("tw, Hb, Bb and Be must be positive")
    if Be > Bb:
        raise GeometryError(f"Be ({Be}) cannot exceed Bb ({Bb})")
    h = Hb / _sin(theta)
    return tw * h * Bb + (tw / 3.0) * Bb**2 - (tw / 3.0) * ((Bb - Be) ** 3 / Bb)


def gapped_K_weld_length(Hb: Inch, Bb: Inch, tb: Inch, theta: Degrees) -> tuple[Inch, str]:
    """Effective weld length for a gapped K-connection branch.

    AISC 360-16, Eqs. K5-8 and K5-9, Table K5.1, p. 16.1-163::

        theta <= 50 deg:  le = 2*(Hb - 1.2*tb)/sin(theta) + 2*(Bb - 1.2*tb)  (K5-8)
        theta >= 60 deg:  le = 2*(Hb - 1.2*tb)/sin(theta) + (Bb - 1.2*tb)    (K5-9)

    with **linear interpolation between 50 and 60 degrees**, which Table K5.1
    mandates and which is the trap here: the two expressions differ by a whole
    ``(Bb - 1.2*tb)`` term, so treating the boundary as a step overstates a
    55-degree joint by up to a third.

    The ``1.2*tb`` deductions remove the corner radii, which carry no weld.
    """
    if min(Hb, Bb, tb) <= 0.0:
        raise GeometryError("Hb, Bb and tb must be positive")
    web = Hb - 1.2 * tb
    flange = Bb - 1.2 * tb
    if min(web, flange) <= 0.0:
        raise GeometryError(
            f"the 1.2*tb corner deduction exceeds the branch dimensions "
            f"(Hb - 1.2tb = {web:.4g}, Bb - 1.2tb = {flange:.4g})"
        )

    s = _sin(theta)
    short = 2.0 * web / s + 2.0 * flange  # Eq. K5-8
    long_ = 2.0 * web / s + flange  # Eq. K5-9

    if theta <= 50.0:
        return short, "Eq. K5-8, theta <= 50 deg"
    if theta >= 60.0:
        return long_, "Eq. K5-9, theta >= 60 deg"
    fraction = (theta - 50.0) / 10.0
    return (
        short + (long_ - short) * fraction,
        f"linear interpolation between Eqs. K5-8 and K5-9 at theta = {theta:g} deg",
    )


def overlapping_branch_weld_length(
    Hbi: Inch, Bbi: Inch, Bei: Inch, Bej: Inch, theta_i: Degrees, theta_j: Degrees, Ov: Ratio
) -> tuple[Inch, str]:
    """Effective weld length of the **overlapping** branch of a K-connection.

    AISC 360-16, Eqs. K5-10, K5-11 and K5-12, Table K5.1, p. 16.1-164. All three
    share the form::

        le,i = 2*[(1 - Ov/100)*(Hbi/sin(theta_i))
                  + (Ov/100)*(Hbi/sin(theta_i + theta_j))] + X

    where ``X`` is ``Bei + Bej`` below 80% overlap (Eqs. K5-10 and K5-11) and
    ``Bbi + Bej`` at 80% and above (Eq. K5-12) -- the same switch from effective
    to full width as Eq. K3-12, and for the same reason.

    Eq. K5-10 additionally scales the whole web term by ``Ov/50``.

    The ``sin(theta_i + theta_j)`` term is what makes the overlapped portion
    different: that length of weld runs along the *other branch*, at the combined
    angle, not along the chord.
    """
    if not 25.0 <= Ov <= 100.0:
        raise AISC360Error(f"overlap Ov = {Ov} is outside the 25%-100% range")
    if min(Hbi, Bbi) <= 0.0:
        raise GeometryError("Hbi and Bbi must be positive")

    combined = theta_i + theta_j
    if not 0.0 < combined <= 180.0:
        raise GeometryError(f"theta_i + theta_j must lie in (0, 180], got {combined}")
    sin_combined = math.sin(math.radians(combined))
    # sin(180 deg) evaluates to 1.22e-16, not 0, so a bare <= 0 test never fires
    # and the division would return a length of order 1e16 instead of raising.
    if sin_combined < 1e-9:
        raise GeometryError(
            f"theta_i + theta_j = {combined:g} deg gives sin = {sin_combined:.4g}: "
            "the branches are collinear and the overlapped weld has no length"
        )

    fraction = Ov / 100.0
    web = (1.0 - fraction) * (Hbi / _sin(theta_i)) + fraction * (Hbi / sin_combined)

    if Ov < 50.0:
        le = (Ov / 50.0) * 2.0 * web + Bei + Bej  # Eq. K5-10
        return le, f"Eq. K5-10, Ov = {Ov:g}%: web term scaled by Ov/50"
    if Ov < 80.0:
        return 2.0 * web + Bei + Bej, f"Eq. K5-11, Ov = {Ov:g}%"  # Eq. K5-11
    return (
        2.0 * web + Bbi + Bej,  # Eq. K5-12
        f"Eq. K5-12, Ov = {Ov:g}%: full Bbi replaces Bei",
    )


def overlapped_branch_weld_length(
    Hbj: Inch, Bej: Inch, tbj: Inch, theta_j: Degrees, *, wide_or_steep: bool = False
) -> tuple[Inch, str]:
    """Effective weld length of the **overlapped** branch of a K-connection.

    AISC 360-16, Eqs. K5-13 and K5-14, Table K5.1, p. 16.1-164::

        le,j = 2*Hbj/sin(theta_j) + 2*Bej                          (K5-13)

        when Bbj/B > 0.85 or theta_j > 50 deg:
        le,j = 2*(Hbj - 1.2*tbj)/sin(theta_j)                      (K5-14)

    Eq. K5-14 **drops the flange contribution entirely** -- not reduces it. A
    wide or steep overlapped branch transfers through its webs alone, because
    its flange is covered by the overlapping branch.
    """
    if min(Hbj, tbj) <= 0.0:
        raise GeometryError("Hbj and tbj must be positive")
    s = _sin(theta_j)
    if wide_or_steep:
        web = Hbj - 1.2 * tbj
        if web <= 0.0:
            raise GeometryError(f"Hbj - 1.2*tbj = {web:.4g} is not positive")
        return 2.0 * web / s, "Eq. K5-14: Bbj/B > 0.85 or theta_j > 50 deg, webs only"
    if Bej <= 0.0:
        raise GeometryError(f"Bej must be positive, got {Bej}")
    return 2.0 * Hbj / s + 2.0 * Bej, "Eq. K5-13"
