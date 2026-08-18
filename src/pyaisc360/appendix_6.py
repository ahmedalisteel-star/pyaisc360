"""Appendix 6 -- Member Stability Bracing.

Covers Sects. 6.1 through 6.4, pp. 16.1-237 to 16.1-244.

Every provision here returns **two** requirements -- a strength and a stiffness
-- and both must be satisfied. A brace strong enough but too flexible does not
brace: the member simply buckles carrying the brace along with it. That is why
:class:`BracingRequirement` always carries the pair.

Three bracing arrangements, distinguished by what they control:

* **panel** (formerly "relative") -- controls the *angular deviation* of a
  segment between braced points, i.e. one end's lateral displacement relative to
  the other. Its requirement is a **shear**.
* **point** (formerly "nodal") -- controls movement at the braced point itself
  without interacting with adjacent points. Its requirement is a **force**.
* **torsional** -- restrains twist rather than lateral displacement, and may be
  attached anywhere on the section, not just near the compression flange.

The resistance factor is **phi = 0.75** throughout -- not the 0.90 of the member
chapters -- except in Eqs. A-6-11a/A-6-11b, where ``Omega = 3.00`` rather than
2.00 because the moment term is squared and the safety factor squares with it.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.citations import Citation, cite
from .core.config import Basis
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.units import Inch, Inch4, Kip, KipIn, Ksi

__all__ = [
    "PHI_BRACING",
    "OMEGA_BRACING",
    "OMEGA_TORSIONAL",
    "CD_DEFAULT",
    "CD_NEAR_INFLECTION",
    "BracingRequirement",
    "column_panel_bracing",
    "column_point_bracing",
    "beam_panel_bracing",
    "beam_point_bracing",
    "beam_torsional_bracing",
    "web_distortional_stiffness",
    "continuous_web_distortional_stiffness",
    "beam_column_bracing",
]

#: Sects. 6.2 and 6.3, pp. 16.1-239 to 16.1-241.
PHI_BRACING: float = 0.75
OMEGA_BRACING: float = 2.00

#: Eqs. A-6-11a/A-6-11b User Note, p. 16.1-242: "Omega = 1.5^2/phi = 3.00 ...
#: because the moment term is squared". The safety factor squares with the
#: quantity it protects -- using 2.00 here would understate the requirement by a
#: third.
OMEGA_TORSIONAL: float = 3.00

#: Sect. 6.3.1a, p. 16.1-241: Cd = 1.0 in general.
CD_DEFAULT: float = 1.0
#: Cd = 2.0 for the brace closest to the inflection point in a beam subject to
#: double curvature bending. That brace works twice as hard because the flange
#: it restrains changes from compression to tension across it.
CD_NEAR_INFLECTION: float = 2.0


@dataclass(frozen=True, slots=True)
class BracingRequirement:
    """A required brace strength **and** stiffness. Both must be satisfied.

    Attributes
    ----------
    strength:
        Required brace strength -- a shear (kips) for panel bracing, a force
        (kips) for point bracing, a moment (kip-in.) for torsional bracing.
    stiffness:
        Required brace stiffness -- kip/in. for lateral bracing, kip-in./rad for
        torsional bracing.
    strength_citation, stiffness_citation:
        The two equations, which are always different.
    kind:
        ``"panel"``, ``"point"`` or ``"torsional"``.
    note:
        Anything qualifying the result.
    """

    strength: float
    stiffness: float
    strength_citation: Citation
    stiffness_citation: Citation
    kind: str
    note: str = ""

    def __post_init__(self) -> None:
        if self.strength < 0.0:
            raise AISC360Error(f"required brace strength cannot be negative, got {self.strength}")

    def report(self, width: int = 78) -> str:
        strength_unit = "kip-in." if self.kind == "torsional" else "kip"
        stiffness_unit = "kip-in./rad" if self.kind == "torsional" else "kip/in."
        lines = ["=" * width, f"Appendix 6 -- {self.kind} bracing requirement", "=" * width]
        lines.append(
            f"{'required strength  (' + self.strength_citation.equation + ')':<48}"
            f"{self.strength:>16,.4g} {strength_unit}"
        )
        lines.append(
            f"{'required stiffness (' + self.stiffness_citation.equation + ')':<48}"
            f"{self.stiffness:>16,.4g} {stiffness_unit}"
        )
        if self.note:
            lines.append(f"    {self.note}")
        lines.append("-" * width)
        lines.append("both must be satisfied -- a strong but flexible brace does not brace")
        lines.append("=" * width)
        return "\n".join(lines)


def _stiffness_factor(basis: Basis, *, torsional: bool = False) -> float:
    """``1/phi`` under LRFD, or ``Omega`` under ASD."""
    if basis.is_lrfd:
        return 1.0 / PHI_BRACING
    return OMEGA_TORSIONAL if torsional else OMEGA_BRACING


# ===========================================================================
# Sect. 6.2 -- Column Bracing
# ===========================================================================
def column_panel_bracing(Pr: Kip, Lbr: Inch, basis: Basis = Basis.LRFD) -> BracingRequirement:
    """Panel bracing of a column, Sect. 6.2.1, p. 16.1-239.

    ::

        Vbr  = 0.005*Pr                                       (Eq. A-6-1)
        beta = (1/phi)*(2*Pr/Lbr)   LRFD                      (Eq. A-6-2a)
        beta = Omega*(2*Pr/Lbr)     ASD                       (Eq. A-6-2b)

    with ``phi = 0.75`` and ``Omega = 2.00``.

    Sect. 6.2.1 also requires the *connection* of the panel bracing system to
    the column to have the strength Sect. 6.2.2 specifies for a **point** brace
    at that location -- a separate and larger force than the panel shear.
    """
    if Pr < 0.0:
        raise GeometryError(f"required axial strength must be non-negative, got {Pr}")
    if Lbr <= 0.0:
        raise GeometryError(f"unbraced length must be positive, got {Lbr}")

    return BracingRequirement(
        strength=0.005 * Pr,  # Eq. A-6-1
        stiffness=_stiffness_factor(basis) * (2.0 * Pr / Lbr),  # Eqs. A-6-2a / A-6-2b
        strength_citation=cite("A-6-1"),
        stiffness_citation=cite("A-6-2a" if basis.is_lrfd else "A-6-2b"),
        kind="panel",
        note=(
            "Sect. 6.2.1: the connection to the panel system must additionally "
            "carry the Sect. 6.2.2 point-brace force at that location"
        ),
    )


def column_point_bracing(Pr: Kip, Lbr: Inch, basis: Basis = Basis.LRFD) -> BracingRequirement:
    """Point bracing of a column, Sect. 6.2.2, p. 16.1-239.

    ::

        Pbr  = 0.01*Pr                                        (Eq. A-6-3)
        beta = (1/phi)*(8*Pr/Lbr)   LRFD                      (Eq. A-6-4a)
        beta = Omega*(8*Pr/Lbr)     ASD                       (Eq. A-6-4b)

    A point brace needs **twice the strength and four times the stiffness** of a
    panel brace over the same length -- it works alone, where panel braces share
    the deviation of a whole segment.

    ``Pr`` is the *largest* of the required axial strengths within the unbraced
    lengths adjacent to the brace, and where those lengths have different
    ``Pr/Lbr`` the larger governs the stiffness.

    Sect. 6.2.2 adds that for **intermediate** point bracing of an individual
    column, ``Lbr`` need not be taken less than the maximum effective length
    permitted for the column at ``Pr`` -- a relief that can matter greatly for a
    lightly loaded column with close braces, since ``beta`` goes as ``1/Lbr``.
    """
    if Pr < 0.0:
        raise GeometryError(f"required axial strength must be non-negative, got {Pr}")
    if Lbr <= 0.0:
        raise GeometryError(f"unbraced length must be positive, got {Lbr}")

    return BracingRequirement(
        strength=0.01 * Pr,  # Eq. A-6-3
        stiffness=_stiffness_factor(basis) * (8.0 * Pr / Lbr),  # Eqs. A-6-4a / A-6-4b
        strength_citation=cite("A-6-3"),
        stiffness_citation=cite("A-6-4a" if basis.is_lrfd else "A-6-4b"),
        kind="point",
        note=(
            "Sect. 6.2.2: for intermediate bracing, Lbr need not be taken less "
            "than the maximum Lc permitted for the column at this Pr"
        ),
    )


# ===========================================================================
# Sect. 6.3.1 -- Beam Lateral Bracing
# ===========================================================================
def beam_panel_bracing(
    Mr: KipIn, ho: Inch, Lbr: Inch, basis: Basis = Basis.LRFD, Cd: float = CD_DEFAULT
) -> BracingRequirement:
    """Panel lateral bracing of a beam, Sect. 6.3.1a, p. 16.1-240.

    ::

        Vbr  = 0.01*(Mr*Cd/ho)                                (Eq. A-6-5)
        beta = (1/phi)*(4*Mr*Cd/(Lbr*ho))   LRFD              (Eq. A-6-6a)
        beta = Omega*(4*Mr*Cd/(Lbr*ho))     ASD               (Eq. A-6-6b)

    ``Mr/ho`` converts the moment into the flange force the brace actually
    resists -- which is why ``ho``, the distance between flange centroids, is
    the divisor rather than the section depth.

    Sect. 6.3 requires lateral bracing to be attached at or near the
    **compression** flange, with two exceptions: at the free end of a cantilever
    it goes at or near the top (tension) flange, and for double-curvature beams
    it must be attached near **both** flanges at the braced point nearest the
    inflection point.

    Sect. 6.3 also warns that in double curvature "the inflection point shall
    not be considered a braced point unless bracing is provided at that
    location" -- treating it as one is a classic way to under-brace a beam.
    """
    if Mr < 0.0:
        raise GeometryError(f"required flexural strength must be non-negative, got {Mr}")
    if ho <= 0.0 or Lbr <= 0.0:
        raise GeometryError(f"ho and Lbr must be positive, got {ho}, {Lbr}")
    if Cd not in (CD_DEFAULT, CD_NEAR_INFLECTION):
        raise AISC360Error(f"Cd is 1.0, or 2.0 near an inflection point; got {Cd}")

    flange_force = Mr * Cd / ho
    return BracingRequirement(
        strength=0.01 * flange_force,  # Eq. A-6-5
        stiffness=_stiffness_factor(basis) * (4.0 * flange_force / Lbr),  # A-6-6a / A-6-6b
        strength_citation=cite("A-6-5"),
        stiffness_citation=cite("A-6-6a" if basis.is_lrfd else "A-6-6b"),
        kind="panel",
        note=f"Cd = {Cd:g}; attach at or near the compression flange (Sect. 6.3.1)",
    )


def beam_point_bracing(
    Mr: KipIn, ho: Inch, Lbr: Inch, basis: Basis = Basis.LRFD, Cd: float = CD_DEFAULT
) -> BracingRequirement:
    """Point lateral bracing of a beam, Sect. 6.3.1b, p. 16.1-241.

    ::

        Pbr  = 0.02*(Mr*Cd/ho)                                (Eq. A-6-7)
        beta = (1/phi)*(10*Mr*Cd/(Lbr*ho))   LRFD             (Eq. A-6-8a)
        beta = Omega*(10*Mr*Cd/(Lbr*ho))     ASD              (Eq. A-6-8b)

    Twice the strength and 2.5 times the stiffness of the panel requirement --
    the same pattern as columns, though the stiffness ratio there is 4.
    """
    if Mr < 0.0:
        raise GeometryError(f"required flexural strength must be non-negative, got {Mr}")
    if ho <= 0.0 or Lbr <= 0.0:
        raise GeometryError(f"ho and Lbr must be positive, got {ho}, {Lbr}")
    if Cd not in (CD_DEFAULT, CD_NEAR_INFLECTION):
        raise AISC360Error(f"Cd is 1.0, or 2.0 near an inflection point; got {Cd}")

    flange_force = Mr * Cd / ho
    return BracingRequirement(
        strength=0.02 * flange_force,  # Eq. A-6-7
        stiffness=_stiffness_factor(basis) * (10.0 * flange_force / Lbr),  # A-6-8a / A-6-8b
        strength_citation=cite("A-6-7"),
        stiffness_citation=cite("A-6-8a" if basis.is_lrfd else "A-6-8b"),
        kind="point",
        note=(
            f"Cd = {Cd:g}; Sect. 6.3.1b: for intermediate bracing, Lbr need not be "
            "taken less than the maximum Lb permitted at this Mr"
        ),
    )


# ===========================================================================
# Sect. 6.3.2 -- Beam Torsional Bracing
# ===========================================================================
def web_distortional_stiffness(
    E: Ksi, ho: Inch, tw: Inch, *, tst: Inch = 0.0, bs: Inch = 0.0
) -> float:
    """Web distortional stiffness ``beta_sec``, including any web stiffener.

    AISC 360-16, Eq. A-6-12, Sect. 6.3.2a, p. 16.1-242::

        beta_sec = (3.3*E/ho)*(1.5*ho*tw^3/12 + tst*bs^3/12)

    A torsional brace restrains twist only if the web itself is stiff enough not
    to distort out of plane under the brace's couple. Without a stiffener the
    ``tw^3`` term alone is often too small, which is what
    :func:`beam_torsional_bracing` reports.

    Parameters
    ----------
    tst, bs:
        Web stiffener thickness and width; ``bs`` is the width for a one-sided
        stiffener, or **twice** the individual width for a pair. Both zero for
        an unstiffened web.
    """
    if ho <= 0.0 or tw <= 0.0:
        raise GeometryError(f"ho and tw must be positive, got {ho}, {tw}")
    return (3.3 * E / ho) * (1.5 * ho * tw**3 / 12.0 + tst * bs**3 / 12.0)


def continuous_web_distortional_stiffness(E: Ksi, ho: Inch, tw: Inch) -> float:
    """Web distortional stiffness for **continuous** torsional bracing.

    AISC 360-16, Eq. A-6-13, Sect. 6.3.2b, p. 16.1-243::

        beta_sec = 3.3*E*tw^3/(12*ho)

    Per unit length, and with no stiffener term -- continuous bracing has no
    discrete points at which to stiffen.
    """
    if ho <= 0.0 or tw <= 0.0:
        raise GeometryError(f"ho and tw must be positive, got {ho}, {tw}")
    return 3.3 * E * tw**3 / (12.0 * ho)


def beam_torsional_bracing(
    Mr: KipIn,
    L: Inch,
    n: int,
    E: Ksi,
    Iyeff: Inch4,
    Cb: float,
    beta_sec: float | None,
    basis: Basis = Basis.LRFD,
) -> BracingRequirement:
    """Point torsional bracing of a beam, Sect. 6.3.2a, pp. 16.1-241 to 16.1-242.

    ::

        Mbr    = 0.02*Mr                                      (Eq. A-6-9)
        beta   = beta_T/(1 - beta_T/beta_sec)                 (Eq. A-6-10)
        beta_T = (1/phi)*(2.4*L/(n*E*Iyeff))*(Mr/Cb)^2  LRFD  (Eq. A-6-11a)
        beta_T = Omega*(2.4*L/(n*E*Iyeff))*(Mr/Cb)^2    ASD   (Eq. A-6-11b)

    with ``phi = 0.75`` and ``Omega = 3.00`` -- **not** 2.00, because the moment
    term is squared (User Note, p. 16.1-242: ``Omega = 1.5^2/phi``).

    ``beta_sec`` may be taken as infinity (pass ``None``) "when a cross-frame is
    attached near both flanges or a vertical diaphragm element is used that is
    approximately the same depth as the beam being braced", giving
    ``beta_br = beta_T``.

    Raises
    ------
    OutOfScopeError
        If ``beta_sec <= beta_T``. Eq. A-6-10 then returns zero or a negative
        number, and the User Note on p. 16.1-243 states what that means:
        "torsional beam bracing will not be effective due to inadequate web
        distortional stiffness". No brace of any stiffness will work until the
        web is stiffened -- so this raises rather than returning a nonsense value.

    Parameters
    ----------
    n:
        Number of braced points within the span.
    Iyeff:
        Effective out-of-plane moment of inertia, ``Iyc + (t/c)*Iyt``. Equal to
        ``Iy`` for a doubly symmetric member, where ``c = t``.
    Cb:
        ``Mr/Cb`` is "the maximum value of the required flexural strength of the
        beam divided by the moment gradient factor" within the adjacent unbraced
        lengths -- so ``Cb`` divides ``Mr`` *before* squaring.
    """
    if Mr < 0.0:
        raise GeometryError(f"required flexural strength must be non-negative, got {Mr}")
    if L <= 0.0 or Iyeff <= 0.0:
        raise GeometryError(f"L and Iyeff must be positive, got {L}, {Iyeff}")
    if n < 1:
        raise GeometryError(f"there must be at least one braced point, got n={n}")
    if Cb <= 0.0:
        raise GeometryError(f"Cb must be positive, got {Cb}")

    factor = _stiffness_factor(basis, torsional=True)
    beta_T = factor * (2.4 * L / (n * E * Iyeff)) * (Mr / Cb) ** 2  # A-6-11a / A-6-11b

    if beta_sec is None:
        beta_br = beta_T
        note = "beta_sec taken as infinity (cross-frame near both flanges)"
    else:
        if beta_sec <= beta_T:
            raise OutOfScopeError(
                f"beta_sec = {beta_sec:.4g} does not exceed beta_T = {beta_T:.4g}. "
                "Eq. A-6-10 is then zero or negative, which the User Note on "
                "p. 16.1-243 identifies as torsional bracing being ineffective due "
                "to inadequate web distortional stiffness. Add a web stiffener."
            )
        beta_br = beta_T / (1.0 - beta_T / beta_sec)  # Eq. A-6-10
        note = f"beta_T = {beta_T:.4g}, beta_sec = {beta_sec:.4g}"

    return BracingRequirement(
        strength=0.02 * Mr,  # Eq. A-6-9
        stiffness=beta_br,
        strength_citation=cite("A-6-9"),
        stiffness_citation=cite("A-6-10"),
        kind="torsional",
        note=note + f"; Omega = {OMEGA_TORSIONAL:g} in Eqs. A-6-11a/A-6-11b (moment is squared)",
    )


# ===========================================================================
# Sect. 6.4 -- Beam-Column Bracing
# ===========================================================================
def beam_column_bracing(
    axial: BracingRequirement, flexural: BracingRequirement
) -> BracingRequirement:
    """Combined bracing requirement for a beam-column, Sect. 6.4, p. 16.1-243.

    Sect. 6.4(a) and (b): the required strength is the **sum** of the axial and
    flexural strengths, and the required stiffness the sum of the stiffnesses --
    Eqs. A-6-1 + A-6-5 and A-6-2 + A-6-6 for panel bracing, or A-6-3 + A-6-7 and
    A-6-4 + A-6-8 for point bracing.

    Sect. 6.4(b) carries an important withdrawal: for beam-columns, ``Lbr``
    "shall be taken as the actual unbraced length; the provisions in Sections
    6.2.2 and 6.3.1b, that Lbr need not be taken less than the maximum permitted
    effective length ... shall not be applied." Both relaxations are off, so the
    caller must have used the true spacing in each part.

    Sect. 6.4(c) and (d) are not arithmetic and are left to the engineer:
    torsional bracing combined with panel or point axial bracing must be
    "combined or distributed in a manner consistent with the resistance provided
    by the element(s) of the actual bracing details", and where the combined
    stress puts **both** flanges in compression, both must be laterally
    restrained.

    Raises
    ------
    AISC360Error
        If the two requirements are of different kinds. Sect. 6.4 sums panel
        with panel and point with point; mixing them is case (c), which has no
        formula.
    """
    if axial.kind != flexural.kind:
        raise AISC360Error(
            f"Sect. 6.4(a) and (b) sum like with like, got {axial.kind!r} and "
            f"{flexural.kind!r}. A torsional flexural brace combined with panel or "
            "point axial bracing is Sect. 6.4(c), which requires engineering "
            "judgement rather than a sum."
        )
    return BracingRequirement(
        strength=axial.strength + flexural.strength,
        stiffness=axial.stiffness + flexural.stiffness,
        strength_citation=axial.strength_citation,
        stiffness_citation=axial.stiffness_citation,
        kind=axial.kind,
        note=(
            "Sect. 6.4: axial + flexural. Lbr must be the ACTUAL unbraced length -- "
            "the Sects. 6.2.2 and 6.3.1b relaxations do not apply to beam-columns"
        ),
    )
