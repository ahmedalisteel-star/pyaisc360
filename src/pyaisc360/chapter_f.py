"""Chapter F -- Design of Members for Flexure.

Covers Sects. F1 through F13, pp. 16.1-45 to 16.1-69. At 108 numbered equations
this is the largest chapter in the Specification, but its shape is regular:
every section evaluates some subset of six limit states and takes the lowest
result.

============  =====  ===  ===  ===  ===  ===  ==============================
Section       Y/CFY  LTB  FLB  WLB  TFY  LLB  Applies to
============  =====  ===  ===  ===  ===  ===  ==============================
F2              *     *                       doubly symm. compact I, channel
F3                    *    *                  compact web, noncompact flange
F4              *     *    *          *       noncompact web, singly symm.
F5              *     *    *          *       slender web
F6              *          *                  I and channels, minor axis
F7              *     *    *    *             square/rect HSS and box
F8              *               *             round HSS
F9              *     *    *    *             tees and double angles
F10             *     *                  *    single angles
F11             *     *                       rectangular bars and rounds
F12             *     *    *                  unsymmetrical shapes
============  =====  ===  ===  ===  ===  ===  ==============================

Two structural facts drive the code's organisation:

* **Cb multiplies the LTB branch and is then capped.** Eqs. F2-2, F4-2, F5-3,
  F7-10 and F11-2 all read ``Mn = Cb*[...] <= Mp``. The cap is not optional --
  without it a large Cb produces a nominal strength above the plastic moment.
  :func:`capped_ltb` applies the pattern once so no section can forget it.
* **The noncompact branches are all the same straight line.** F3-1, F4-13,
  F6-2, F7-2, F7-6, F9-14 and F10-6 interpolate between a plastic and an
  elastic anchor, either on lambda or directly on a width-to-thickness ratio.
  The lambda-based ones go through :func:`pyaisc360.utils.linear_transition`.

Section F13 is not a strength section: it is a set of proportioning limits and
a tension-rupture cap that overlays whatever F2-F12 produced.

Eq. F2-7 (``rts``) and Eqs. F2-8a/F2-8b (``c``) are section *properties*, not
strength provisions, so they live with the section adapter as
:func:`pyaisc360.sections.derive_rts` and
:func:`pyaisc360.sections.derive_c_channel` and are passed in here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from .chapter_b import kc_coefficient
from .core.citations import cite
from .core.config import DEFAULT_SETTINGS, Basis, DesignSettings
from .core.enums import Axis, FlexuralSlenderness, LimitState
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.result import LimitStateResult, StrengthResult
from .core.units import Inch, Inch2, Inch3, Inch4, KipIn, Ksi, Ratio
from .materials import Steel
from .sections import require_properties
from .utils import linear_transition

__all__ = [
    "PHI_B",
    "OMEGA_B",
    "CB_ANGLE_MAX",
    "ShapeType",
    "FlexuralMember",
    "plastic_moment",
    "yield_moment",
    "capped_ltb",
    # F2
    "Lp_F2",
    "Lr_F2",
    "Fcr_F2",
    "f2_strength",
    # F3
    "f3_strength",
    # F4
    "web_plastification_factor",
    "FL_stress",
    "rt_effective_radius",
    "aw_ratio",
    "Lp_F4",
    "Lr_F4",
    "Fcr_F4",
    "f4_strength",
    # F5
    "Rpg_factor",
    "Lr_F5",
    "f5_strength",
    # F6
    "f6_strength",
    # F7
    "hss_effective_flange_width",
    "effective_section_modulus_rect_hss",
    "Lp_F7",
    "Lr_F7",
    "f7_strength",
    # F8
    "f8_strength",
    # F9
    "Lp_F9",
    "Lr_F9",
    "Mcr_F9",
    "f9_strength",
    # F10
    "Mcr_F10_principal",
    "Mcr_F10_geometric",
    "f10_strength",
    # F11
    "f11_strength",
    # F12
    "f12_strength",
    # F13
    "tension_rupture_cap",
    "check_proportioning_limits",
    "cover_plate_extension",
    "Lm_moment_redistribution",
    "web_slenderness_limit",
    # orchestrator
    "flexural_strength",
]

#: Resistance and safety factors for flexure. AISC 360-16, Sect. F1(a), p. 16.1-46.
#: "For all provisions in this chapter" -- there is no per-section variation.
PHI_B: float = 0.90
OMEGA_B: float = 1.67

#: Sect. F10.2, p. 16.1-63: for single angles, Cb from Eq. F1-1 is capped at 1.5.
#: This cap exists only in F10; no other section limits Cb at all.
CB_ANGLE_MAX: float = 1.5

#: Sect. F6.1, F9.2 and F11.1 cap the plastic moment at 1.6*Fy*S. The bound stops
#: the shape factor of a stocky section implying a moment the elastic core cannot
#: deliver before excessive yielding. It appears in Eqs. F6-1, F9-2, F11-1 and in
#: the Mp definition used by Eqs. F4-9b and F4-16b.
SHAPE_FACTOR_CAP: float = 1.6


class ShapeType(str, Enum):
    """Cross-section family -- selects which Chapter F section applies."""

    ROLLED_I = "rolled I-shape"
    BUILT_UP_I = "built-up I-shape"
    CHANNEL = "channel"
    RECTANGULAR_HSS = "rectangular or square HSS"
    BOX = "box section"
    ROUND_HSS = "round HSS"
    TEE = "tee"
    DOUBLE_ANGLE = "double angle"
    SINGLE_ANGLE = "single angle"
    RECTANGULAR_BAR = "rectangular bar"
    ROUND_BAR = "round bar"
    UNSYMMETRICAL = "unsymmetrical shape"


@dataclass(frozen=True)
class FlexuralMember:
    """A member in flexure.

    Attributes
    ----------
    section:
        Anything satisfying the Wave 0 section contract.
    steel:
        Grade, supplying ``Fy``, ``Fu`` and ``E``.
    shape:
        Which cross-section family -- drives the section selection.
    Lb:
        Laterally unbraced length, in.: "length between points that are either
        braced against lateral displacement of the compression flange or braced
        against twist of the cross section" (Sect. F2.2, p. 16.1-47). Zero means
        continuously braced, and LTB does not apply.
    Cb:
        Lateral-torsional buckling modification factor from Eq. F1-1. Compute it
        with :func:`pyaisc360.utils.lateral_torsional_modification_factor`;
        1.0 is always permitted and always conservative.
    axis:
        Bending axis. ``Axis.MINOR`` sends I-shapes and channels to Sect. F6.
    continuous_restraint:
        Single angles only -- whether there is continuous lateral-torsional
        restraint along the length (Sect. F10, p. 16.1-62).
    tension_flange_in_compression:
        Tees and double angles only -- whether the stem or web leg is in
        compression, which changes the sign of ``B`` in Eqs. F9-11/F9-12.
    """

    section: Any
    steel: Steel
    shape: ShapeType = ShapeType.ROLLED_I
    Lb: Inch = 0.0
    Cb: float = 1.0
    axis: Axis = Axis.MAJOR
    continuous_restraint: bool = False
    stem_in_compression: bool = False

    def __post_init__(self) -> None:
        if self.Lb < 0.0:
            raise GeometryError(f"unbraced length must be non-negative, got {self.Lb}")
        if self.Cb <= 0.0:
            raise GeometryError(f"Cb must be positive, got {self.Cb}")


# ===========================================================================
# Shared building blocks
# ===========================================================================
def plastic_moment(Fy: Ksi, Z: Inch3, *, S: Inch3 | None = None) -> KipIn:
    """Plastic moment ``Mp = Fy*Z``, optionally capped at ``1.6*Fy*S``.

    AISC 360-16, Eq. F2-1 (uncapped) and Eqs. F6-1, F9-2, F11-1 (capped).

    Pass ``S`` wherever the governing section imposes the ``1.6*Fy*S`` bound;
    omit it for Sects. F2, F7 and F8, which do not.
    """
    if Fy <= 0.0 or Z <= 0.0:
        raise GeometryError(f"Fy and Z must be positive, got Fy={Fy}, Z={Z}")
    Mp = Fy * Z
    if S is not None:
        if S <= 0.0:
            raise GeometryError(f"S must be positive, got {S}")
        return min(Mp, SHAPE_FACTOR_CAP * Fy * S)
    return Mp


def yield_moment(Fy: Ksi, S: Inch3) -> KipIn:
    """Yield moment ``My = Fy*S``.

    AISC 360-16, Eq. F9-3, Sect. F9.1, p. 16.1-60; the same product appears as
    ``Myc = Fy*Sxc`` in Eq. F4-4 and ``Myt = Fy*Sxt`` in Sect. F4.4.
    """
    if Fy <= 0.0 or S <= 0.0:
        raise GeometryError(f"Fy and S must be positive, got Fy={Fy}, S={S}")
    return Fy * S


def capped_ltb(Cb: float, bracketed: KipIn, cap: KipIn) -> KipIn:
    """Apply ``Mn = Cb * [...] <= cap`` -- the shared shape of the LTB branches.

    AISC 360-16, Eqs. F2-2, F4-2, F7-10 (inelastic LTB) and F2-3, F4-3, F7-11
    (elastic LTB).

    The cap is the whole point. ``Cb`` can legitimately exceed 2.2, and without
    the ``<= Mp`` bound the inelastic branch would return a nominal strength
    above the plastic moment at short unbraced lengths -- a strength the section
    cannot develop by any mechanism.
    """
    if Cb <= 0.0:
        raise GeometryError(f"Cb must be positive, got {Cb}")
    return min(Cb * bracketed, cap)


# ===========================================================================
# Sect. F2 -- Doubly Symmetric Compact I-Shapes and Channels, Major Axis
# ===========================================================================
def Lp_F2(ry: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """Limiting unbraced length for yielding, ``Lp``.

    AISC 360-16, Eq. F2-5, Sect. F2.2, p. 16.1-48::

        Lp = 1.76*ry*sqrt(E/Fy)
    """
    if ry <= 0.0:
        raise GeometryError(f"ry must be positive, got {ry}")
    return 1.76 * ry * math.sqrt(E / Fy)


def Lr_F2(rts: Inch, E: Ksi, Fy: Ksi, J: Inch4, c: Ratio, Sx: Inch3, ho: Inch) -> Inch:
    """Limiting unbraced length for inelastic LTB, ``Lr``.

    AISC 360-16, Eq. F2-6, Sect. F2.2, p. 16.1-48::

        Lr = 1.95*rts*(E/(0.7*Fy))
             * sqrt( J*c/(Sx*ho) + sqrt( (J*c/(Sx*ho))^2 + 6.76*(0.7*Fy/E)^2 ) )

    Parameters
    ----------
    c:
        1.0 for a doubly symmetric I-shape (Eq. F2-8a); for a channel, from
        Eq. F2-8b -- see :func:`pyaisc360.sections.derive_c_channel`.

    Notes
    -----
    ``Lr`` is by definition the length at which Eq. F2-4 returns ``Fcr = 0.7*Fy``,
    so Eqs. F2-2 and F2-3 ought to meet exactly at ``Lb = Lr``. They do not,
    because both of Eq. F2-6's coefficients are rounded::

        printed 1.95  ->  sqrt(pi^4*0.078/2)    = 1.949091   (0.047% high)
        printed 6.76  ->  4/(pi^4*0.078^2)      = 6.749495   (0.156% high)

    With the exact coefficients ``Fcr(Lr) = 0.7*Fy`` to 2e-16; with the printed
    ones there is a **0.12% step** between the two branches at ``Lb = Lr``. The
    printed values are used here -- they are what the Specification mandates and
    what the Manual's tabulated ``Lr`` reflects -- but no test should expect the
    branches to agree to machine precision. This is the same class of rounding as
    the 4.71 of Sect. E3.
    """
    for name, value in (("rts", rts), ("Sx", Sx), ("ho", ho), ("c", c)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    if J < 0.0:
        raise GeometryError(f"J must be non-negative, got {J}")

    term = J * c / (Sx * ho)
    residual = 6.76 * (0.7 * Fy / E) ** 2
    return 1.95 * rts * (E / (0.7 * Fy)) * math.sqrt(term + math.sqrt(term**2 + residual))


def Fcr_F2(Cb: float, Lb: Inch, rts: Inch, E: Ksi, J: Inch4, c: Ratio, Sx: Inch3, ho: Inch) -> Ksi:
    """Elastic LTB critical stress.

    AISC 360-16, Eq. F2-4, Sect. F2.2, p. 16.1-47::

        Fcr = Cb*pi^2*E/(Lb/rts)^2 * sqrt(1 + 0.078*J*c/(Sx*ho)*(Lb/rts)^2)

    The User Note on p. 16.1-48 permits the square-root term to be taken as 1.0
    conservatively. It is evaluated here -- the saving is real, up to 40% for a
    stocky rolled shape at moderate ``Lb``.
    """
    if Lb <= 0.0:
        raise GeometryError(f"Lb must be positive for elastic LTB, got {Lb}")
    if rts <= 0.0 or Sx <= 0.0 or ho <= 0.0:
        raise GeometryError(f"rts, Sx and ho must be positive, got {rts}, {Sx}, {ho}")

    slenderness = Lb / rts
    warping = 1.0 + 0.078 * J * c / (Sx * ho) * slenderness**2
    return Cb * math.pi**2 * E / slenderness**2 * math.sqrt(warping)


def f2_strength(
    member: FlexuralMember, *, c: Ratio = 1.0
) -> list[LimitStateResult]:
    """Sect. F2 -- yielding and lateral-torsional buckling.

    AISC 360-16, Sect. F2, pp. 16.1-47 to 16.1-48. Applies to doubly symmetric
    I-shapes and channels bent about their major axis with compact webs and
    compact flanges.
    """
    E, Fy = member.steel.E, member.steel.Fy
    props = require_properties(member.section, "Zx", "Sx", "ry", "rts", "J", "ho")
    Zx, Sx, ry, rts, J, ho = (
        props["Zx"], props["Sx"], props["ry"], props["rts"], props["J"], props["ho"]
    )

    Mp = plastic_moment(Fy, Zx)  # Eq. F2-1
    states = [
        LimitStateResult(
            LimitState.PLASTIC_MOMENT, Mp, cite("F2-1"),
            detail={"Fy": Fy, "Zx": Zx}, note="Mp = Fy*Zx",
        )
    ]

    Lp = Lp_F2(ry, E, Fy)  # Eq. F2-5
    Lr = Lr_F2(rts, E, Fy, J, c, Sx, ho)  # Eq. F2-6
    Lb = member.Lb
    shared = {"Lb": Lb, "Lp": Lp, "Lr": Lr, "Cb": member.Cb}

    if Lb <= Lp:
        # Sect. F2.2(a): "the limit state of lateral-torsional buckling does not apply."
        return states

    if Lb <= Lr:
        bracketed = Mp - (Mp - 0.7 * Fy * Sx) * (Lb - Lp) / (Lr - Lp)
        Mn = capped_ltb(member.Cb, bracketed, Mp)  # Eq. F2-2
        states.append(
            LimitStateResult(
                LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, cite("F2-2"),
                detail=shared, note="Lp < Lb <= Lr, inelastic LTB",
            )
        )
    else:
        Fcr = Fcr_F2(member.Cb, Lb, rts, E, J, c, Sx, ho)  # Eq. F2-4
        Mn = min(Fcr * Sx, Mp)  # Eq. F2-3
        states.append(
            LimitStateResult(
                LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, cite("F2-3"),
                detail={**shared, "Fcr": Fcr}, note="Lb > Lr, elastic LTB",
            )
        )
    return states


# ===========================================================================
# Sect. F3 -- Compact Webs, Noncompact or Slender Flanges, Major Axis
# ===========================================================================
def f3_strength(
    member: FlexuralMember,
    *,
    flange_class: FlexuralSlenderness,
    lam: Ratio,
    lam_pf: Ratio,
    lam_rf: Ratio,
    h_over_tw: Ratio,
    c: Ratio = 1.0,
) -> list[LimitStateResult]:
    """Sect. F3 -- LTB per F2.2, plus compression flange local buckling.

    AISC 360-16, Sect. F3, pp. 16.1-49 to 16.1-50.

    Sect. F3.1 says only "for lateral-torsional buckling, the provisions of
    Section F2.2 shall apply", so the LTB branch here *is* Sect. F2's -- it is
    not restated, it is reused.
    """
    E, Fy = member.steel.E, member.steel.Fy
    props = require_properties(member.section, "Zx", "Sx")
    Zx, Sx = props["Zx"], props["Sx"]
    Mp = plastic_moment(Fy, Zx)

    # Sect. F3.1: LTB is Sect. F2.2 verbatim. Drop F2's yielding state -- Sect. F3
    # lists only LTB and FLB, because a noncompact flange yields before Mp.
    states = [s for s in f2_strength(member, c=c)
              if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING]

    if flange_class is FlexuralSlenderness.COMPACT:
        raise OutOfScopeError(
            "Sect. F3 applies to noncompact or slender flanges; a compact "
            "flange with a compact web belongs to Sect. F2"
        )

    if flange_class is FlexuralSlenderness.NONCOMPACT:
        Mn = linear_transition(Mp, 0.7 * Fy * Sx, lam, lam_pf, lam_rf)  # Eq. F3-1
        citation, note = cite("F3-1"), "noncompact flange"
    else:
        kc = kc_coefficient(h_over_tw)
        Mn = 0.9 * E * kc * Sx / lam**2  # Eq. F3-2
        citation, note = cite("F3-2"), f"slender flange, kc = {kc:.3f}"

    states.append(
        LimitStateResult(
            LimitState.FLANGE_LOCAL_BUCKLING, Mn, citation,
            detail={"lambda": lam, "lambda_pf": lam_pf, "lambda_rf": lam_rf, "Mp": Mp},
            note=note,
        )
    )
    return states


# ===========================================================================
# Sect. F4 -- Other I-Shapes, Compact or Noncompact Webs, Major Axis
# ===========================================================================
def FL_stress(Fy: Ksi, Sxt: Inch3, Sxc: Inch3) -> Ksi:
    """Nominal compression flange stress above which inelastic buckling applies.

    AISC 360-16, Eqs. F4-6a and F4-6b, Sect. F4.2(3), p. 16.1-51::

        Sxt/Sxc >= 0.7:  FL = 0.7*Fy                       (Eq. F4-6a)
        Sxt/Sxc <  0.7:  FL = Fy*Sxt/Sxc  >= 0.5*Fy        (Eq. F4-6b)

    For a doubly symmetric shape ``Sxt = Sxc`` and this is always ``0.7*Fy`` --
    which is where the 0.7 in Eqs. F2-2 and F2-6 comes from.
    """
    if Sxt <= 0.0 or Sxc <= 0.0:
        raise GeometryError(f"Sxt and Sxc must be positive, got {Sxt}, {Sxc}")
    if Sxt / Sxc >= 0.7:
        return 0.7 * Fy  # Eq. F4-6a
    return max(Fy * Sxt / Sxc, 0.5 * Fy)  # Eq. F4-6b


def aw_ratio(hc: Inch, tw: Inch, bfc: Inch, tfc: Inch) -> Ratio:
    """Ratio of web area to compression flange area, ``aw``.

    AISC 360-16, Eq. F4-12, Sect. F4.2(7), p. 16.1-53::

        aw = hc*tw / (bfc*tfc)

    Sect. F5 additionally caps ``aw`` at 10 when it is used in Eq. F5-6; that
    cap is applied in :func:`Rpg_factor`, not here, because Eq. F4-11 has no cap.
    """
    if min(hc, tw, bfc, tfc) <= 0.0:
        raise GeometryError(f"hc, tw, bfc and tfc must be positive, got {hc}, {tw}, {bfc}, {tfc}")
    return hc * tw / (bfc * tfc)


def rt_effective_radius(bfc: Inch, aw: Ratio) -> Inch:
    """Effective radius of gyration for LTB, ``rt``.

    AISC 360-16, Eq. F4-11, Sect. F4.2(7)(i), p. 16.1-52::

        rt = bfc / sqrt(12*(1 + aw/6))

    For I-shapes with a rectangular compression flange. Sect. F4.2(7)(ii)
    covers channel caps and cover plates by definition rather than by formula,
    and is left to the engineer.
    """
    if bfc <= 0.0:
        raise GeometryError(f"bfc must be positive, got {bfc}")
    if aw < 0.0:
        raise GeometryError(f"aw must be non-negative, got {aw}")
    return bfc / math.sqrt(12.0 * (1.0 + aw / 6.0))


def web_plastification_factor(
    Mp: KipIn,
    Myc: KipIn,
    lam_w: Ratio,
    lam_pw: Ratio,
    lam_rw: Ratio,
    Iyc_over_Iy: Ratio,
) -> Ratio:
    """Web plastification factor ``Rpc`` (or ``Rpt`` with the tension-flange inputs).

    AISC 360-16, Eqs. F4-9a, F4-9b and F4-10, Sect. F4.2(6), p. 16.1-52.
    The identical structure serves ``Rpt`` as Eqs. F4-16a, F4-16b and F4-17,
    Sect. F4.4, p. 16.1-53 -- pass ``Myt`` in place of ``Myc``::

        Iyc/Iy <= 0.23:            Rpc = 1.0                        (F4-10)
        Iyc/Iy >  0.23, hc/tw <= lambda_pw:  Rpc = Mp/Myc           (F4-9a)
        Iyc/Iy >  0.23, hc/tw >  lambda_pw:
            Rpc = [Mp/Myc - (Mp/Myc - 1)*(lam - lam_pw)/(lam_rw - lam_pw)]
                  <= Mp/Myc                                          (F4-9b)

    The ``Iyc/Iy <= 0.23`` cut-off is the same threshold that forces ``J = 0``
    in Eq. F4-5: below it the compression flange is too small to restrain the
    web, so no plastification credit is taken.
    """
    if Myc <= 0.0:
        raise GeometryError(f"yield moment must be positive, got {Myc}")
    if Iyc_over_Iy <= 0.23:
        return 1.0  # Eq. F4-10 / F4-17

    shape_factor = Mp / Myc
    if lam_w <= lam_pw:
        return shape_factor  # Eq. F4-9a / F4-16a

    # Eq. F4-9b / F4-16b -- interpolates the shape factor down towards 1.0.
    Rpc = shape_factor - (shape_factor - 1.0) * (lam_w - lam_pw) / (lam_rw - lam_pw)
    return min(Rpc, shape_factor)


def Lp_F4(rt: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """Limiting unbraced length for yielding, ``Lp``.

    AISC 360-16, Eq. F4-7, Sect. F4.2(4), p. 16.1-51::

        Lp = 1.1*rt*sqrt(E/Fy)

    Note the coefficient is 1.1 against ``rt``, not the 1.76 against ``ry`` of
    Eq. F2-5. The two are different measures and the coefficients are not
    interchangeable.
    """
    if rt <= 0.0:
        raise GeometryError(f"rt must be positive, got {rt}")
    return 1.1 * rt * math.sqrt(E / Fy)


def Lr_F4(rt: Inch, E: Ksi, FL: Ksi, J: Inch4, Sxc: Inch3, ho: Inch) -> Inch:
    """Limiting unbraced length for inelastic LTB, ``Lr``.

    AISC 360-16, Eq. F4-8, Sect. F4.2(5), p. 16.1-51::

        Lr = 1.95*rt*(E/FL)
             * sqrt( J/(Sxc*ho) + sqrt( (J/(Sxc*ho))^2 + 6.76*(FL/E)^2 ) )

    Structurally Eq. F2-6 with ``FL`` in place of ``0.7*Fy``, ``rt`` for
    ``rts``, ``Sxc`` for ``Sx``, and no ``c`` -- Sect. F4 has no channels.
    """
    if rt <= 0.0 or Sxc <= 0.0 or ho <= 0.0:
        raise GeometryError(f"rt, Sxc and ho must be positive, got {rt}, {Sxc}, {ho}")
    if FL <= 0.0:
        raise GeometryError(f"FL must be positive, got {FL}")

    term = J / (Sxc * ho)
    return 1.95 * rt * (E / FL) * math.sqrt(term + math.sqrt(term**2 + 6.76 * (FL / E) ** 2))


def Fcr_F4(Cb: float, Lb: Inch, rt: Inch, E: Ksi, J: Inch4, Sxc: Inch3, ho: Inch) -> Ksi:
    """Elastic LTB critical stress.

    AISC 360-16, Eq. F4-5, Sect. F4.2(2), p. 16.1-51::

        Fcr = Cb*pi^2*E/(Lb/rt)^2 * sqrt(1 + 0.078*J/(Sxc*ho)*(Lb/rt)^2)

    Sect. F4.2(2) requires ``J`` to be taken as **zero** when
    ``Iyc/Iy <= 0.23``; the caller applies that before calling, because the
    same condition also selects Eq. F4-10 for ``Rpc``.
    """
    if Lb <= 0.0 or rt <= 0.0:
        raise GeometryError(f"Lb and rt must be positive, got {Lb}, {rt}")
    slenderness = Lb / rt
    return (
        Cb * math.pi**2 * E / slenderness**2
        * math.sqrt(1.0 + 0.078 * J / (Sxc * ho) * slenderness**2)
    )


def f4_strength(
    member: FlexuralMember,
    *,
    Sxc: Inch3,
    Sxt: Inch3,
    hc: Inch,
    bfc: Inch,
    tfc: Inch,
    tw: Inch,
    Iyc_over_Iy: Ratio,
    flange_class: FlexuralSlenderness,
    lam_f: Ratio,
    lam_pf: Ratio,
    lam_rf: Ratio,
    lam_w: Ratio,
    lam_pw: Ratio,
    lam_rw: Ratio,
    ho: Inch,
) -> list[LimitStateResult]:
    """Sect. F4 -- compression flange yielding, LTB, FLB and tension flange yielding.

    AISC 360-16, Sect. F4, pp. 16.1-50 to 16.1-54. Applies to doubly symmetric
    I-shapes with noncompact webs, and to singly symmetric I-shapes with webs
    attached at the mid-width of the flanges and compact or noncompact webs.
    """
    E, Fy = member.steel.E, member.steel.Fy
    props = require_properties(member.section, "Zx", "J")
    Zx, J = props["Zx"], props["J"]

    # Sect. F4.2(6): Mp for Rpc carries the 1.6*Fy*Sx cap.
    Mp = plastic_moment(Fy, Zx, S=min(Sxc, Sxt))
    Myc = yield_moment(Fy, Sxc)  # Eq. F4-4
    Myt = yield_moment(Fy, Sxt)
    FL = FL_stress(Fy, Sxt, Sxc)  # Eqs. F4-6a / F4-6b
    Rpc = web_plastification_factor(Mp, Myc, lam_w, lam_pw, lam_rw, Iyc_over_Iy)
    aw = aw_ratio(hc, tw, bfc, tfc)  # Eq. F4-12
    rt = rt_effective_radius(bfc, aw)  # Eq. F4-11

    # Sect. F4.2(2): J is taken as zero when Iyc/Iy <= 0.23.
    J_ltb = 0.0 if Iyc_over_Iy <= 0.23 else J

    states: list[LimitStateResult] = [
        LimitStateResult(
            LimitState.COMPRESSION_FLANGE_YIELDING, Rpc * Myc, cite("F4-1"),
            detail={"Rpc": Rpc, "Myc": Myc, "Mp": Mp, "FL": FL},
            note="Mn = Rpc*Myc",
        )
    ]

    # -- LTB, Sect. F4.2 ---------------------------------------------------
    Lp = Lp_F4(rt, E, Fy)  # Eq. F4-7
    Lr = Lr_F4(rt, E, FL, J_ltb, Sxc, ho)  # Eq. F4-8
    Lb = member.Lb
    if Lb > Lp:
        cap = Rpc * Myc
        if Lb <= Lr:
            bracketed = cap - (cap - FL * Sxc) * (Lb - Lp) / (Lr - Lp)
            Mn = capped_ltb(member.Cb, bracketed, cap)  # Eq. F4-2
            citation, note = cite("F4-2"), "Lp < Lb <= Lr, inelastic LTB"
            extra: dict[str, float] = {}
        else:
            Fcr = Fcr_F4(member.Cb, Lb, rt, E, J_ltb, Sxc, ho)  # Eq. F4-5
            Mn = min(Fcr * Sxc, cap)  # Eq. F4-3
            citation, note = cite("F4-3"), "Lb > Lr, elastic LTB"
            extra = {"Fcr": Fcr}
        states.append(
            LimitStateResult(
                LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, citation,
                detail={"Lb": Lb, "Lp": Lp, "Lr": Lr, "rt": rt, "aw": aw, **extra},
                note=note,
            )
        )

    # -- Compression flange local buckling, Sect. F4.3 ---------------------
    if flange_class is not FlexuralSlenderness.COMPACT:
        if flange_class is FlexuralSlenderness.NONCOMPACT:
            Mn = linear_transition(Rpc * Myc, FL * Sxc, lam_f, lam_pf, lam_rf)  # Eq. F4-13
            citation, note = cite("F4-13"), "noncompact flange"
        else:
            kc = kc_coefficient(lam_w)
            Mn = 0.9 * E * kc * Sxc / lam_f**2  # Eq. F4-14
            citation, note = cite("F4-14"), f"slender flange, kc = {kc:.3f}"
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Mn, citation,
                detail={"lambda": lam_f, "lambda_pf": lam_pf, "lambda_rf": lam_rf},
                note=note,
            )
        )

    # -- Tension flange yielding, Sect. F4.4 -------------------------------
    if Sxt < Sxc:
        Rpt = web_plastification_factor(Mp, Myt, lam_w, lam_pw, lam_rw, Iyc_over_Iy)
        states.append(
            LimitStateResult(
                LimitState.TENSION_FLANGE_YIELDING, Rpt * Myt, cite("F4-15"),
                detail={"Rpt": Rpt, "Myt": Myt},
                note="Sxt < Sxc, so tension flange yielding applies",
            )
        )
    return states


# ===========================================================================
# Sect. F5 -- Slender Webs, Major Axis
# ===========================================================================
def Rpg_factor(aw: Ratio, hc: Inch, tw: Inch, E: Ksi, Fy: Ksi) -> Ratio:
    """Bending strength reduction factor ``Rpg``.

    AISC 360-16, Eq. F5-6, Sect. F5.2, p. 16.1-55::

        Rpg = 1 - aw/(1200 + 300*aw) * (hc/tw - 5.7*sqrt(E/Fy))  <= 1.0

    ``aw`` is Eq. F4-12 "but shall not exceed 10" -- capped here rather than in
    :func:`aw_ratio`, because Eq. F4-11 applies no such cap.

    Also clamped below at zero: a web far past ``5.7*sqrt(E/Fy)`` would drive
    the expression negative, and a negative strength reduction is meaningless.
    Sect. F13.2 independently forbids such a web (Eqs. F13-3/F13-4).
    """
    if hc <= 0.0 or tw <= 0.0:
        raise GeometryError(f"hc and tw must be positive, got {hc}, {tw}")
    aw_capped = min(aw, 10.0)
    excess = hc / tw - 5.7 * math.sqrt(E / Fy)
    Rpg = 1.0 - aw_capped / (1200.0 + 300.0 * aw_capped) * excess
    return min(max(Rpg, 0.0), 1.0)


def Lr_F5(rt: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """Limiting unbraced length for inelastic LTB, slender-web members.

    AISC 360-16, Eq. F5-5, Sect. F5.2, p. 16.1-55::

        Lr = pi*rt*sqrt(E/(0.7*Fy))

    Much simpler than Eq. F4-8 because a slender web contributes no torsional
    restraint worth crediting -- there is no ``J`` term at all.
    """
    if rt <= 0.0:
        raise GeometryError(f"rt must be positive, got {rt}")
    return math.pi * rt * math.sqrt(E / (0.7 * Fy))


def f5_strength(
    member: FlexuralMember,
    *,
    Sxc: Inch3,
    Sxt: Inch3,
    hc: Inch,
    bfc: Inch,
    tfc: Inch,
    tw: Inch,
    flange_class: FlexuralSlenderness,
    lam_f: Ratio,
    lam_pf: Ratio,
    lam_rf: Ratio,
    lam_w: Ratio,
) -> list[LimitStateResult]:
    """Sect. F5 -- slender-web I-shapes: CFY, LTB, FLB and TFY.

    AISC 360-16, Sect. F5, pp. 16.1-54 to 16.1-55.

    Every limit state here is a *stress* multiplied by ``Rpg*Sxc``, not a
    moment interpolation -- that is the structural difference from Sect. F4.
    """
    E, Fy = member.steel.E, member.steel.Fy
    aw = aw_ratio(hc, tw, bfc, tfc)  # Eq. F4-12
    rt = rt_effective_radius(bfc, aw)  # Eq. F4-11
    Rpg = Rpg_factor(aw, hc, tw, E, Fy)  # Eq. F5-6

    states: list[LimitStateResult] = [
        LimitStateResult(
            LimitState.COMPRESSION_FLANGE_YIELDING, Rpg * Fy * Sxc, cite("F5-1"),
            detail={"Rpg": Rpg, "aw": aw, "Sxc": Sxc}, note="Mn = Rpg*Fy*Sxc",
        )
    ]

    # -- LTB, Sect. F5.2 ---------------------------------------------------
    Lp = Lp_F4(rt, E, Fy)  # Eq. F4-7, referenced by Sect. F5.2
    Lr = Lr_F5(rt, E, Fy)  # Eq. F5-5
    Lb = member.Lb
    if Lb > Lp:
        if Lb <= Lr:
            Fcr = min(
                member.Cb * (Fy - 0.3 * Fy * (Lb - Lp) / (Lr - Lp)), Fy
            )  # Eq. F5-3
            citation, note = cite("F5-3"), "Lp < Lb <= Lr, inelastic LTB"
        else:
            Fcr = min(member.Cb * math.pi**2 * E / (Lb / rt) ** 2, Fy)  # Eq. F5-4
            citation, note = cite("F5-4"), "Lb > Lr, elastic LTB"
        states.append(
            LimitStateResult(
                LimitState.LATERAL_TORSIONAL_BUCKLING, Rpg * Fcr * Sxc, citation,
                detail={"Lb": Lb, "Lp": Lp, "Lr": Lr, "Fcr": Fcr, "Rpg": Rpg},
                note=f"{note}; Mn = Rpg*Fcr*Sxc per Eq. F5-2",
            )
        )

    # -- Compression flange local buckling, Sect. F5.3 ---------------------
    if flange_class is not FlexuralSlenderness.COMPACT:
        if flange_class is FlexuralSlenderness.NONCOMPACT:
            Fcr = linear_transition(Fy, 0.7 * Fy, lam_f, lam_pf, lam_rf)  # Eq. F5-8
            citation, note = cite("F5-8"), "noncompact flange"
        else:
            kc = kc_coefficient(lam_w)
            Fcr = 0.9 * E * kc / lam_f**2  # Eq. F5-9
            citation, note = cite("F5-9"), f"slender flange, kc = {kc:.3f}"
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Rpg * Fcr * Sxc, citation,
                detail={"Fcr": Fcr, "Rpg": Rpg, "lambda": lam_f},
                note=f"{note}; Mn = Rpg*Fcr*Sxc per Eq. F5-7",
            )
        )

    # -- Tension flange yielding, Sect. F5.4 -------------------------------
    if Sxt < Sxc:
        states.append(
            LimitStateResult(
                LimitState.TENSION_FLANGE_YIELDING, Fy * Sxt, cite("F5-10"),
                detail={"Sxt": Sxt},
                note="Sxt < Sxc; note Rpg does not appear in Eq. F5-10",
            )
        )
    return states


# ===========================================================================
# Sect. F6 -- I-Shapes and Channels, Minor Axis
# ===========================================================================
def f6_strength(
    member: FlexuralMember,
    *,
    flange_class: FlexuralSlenderness,
    lam: Ratio,
    lam_pf: Ratio,
    lam_rf: Ratio,
) -> list[LimitStateResult]:
    """Sect. F6 -- yielding and flange local buckling about the minor axis.

    AISC 360-16, Sect. F6, p. 16.1-56.

    There is no lateral-torsional buckling limit state: a member already bent
    about its weak axis has nothing weaker to buckle into.
    """
    E, Fy = member.steel.E, member.steel.Fy
    props = require_properties(member.section, "Zy", "Sy")
    Zy, Sy = props["Zy"], props["Sy"]

    Mp = plastic_moment(Fy, Zy, S=Sy)  # Eq. F6-1, with the 1.6*Fy*Sy cap
    states = [
        LimitStateResult(
            LimitState.PLASTIC_MOMENT, Mp, cite("F6-1"),
            detail={"Zy": Zy, "Sy": Sy}, note="Mp = Fy*Zy <= 1.6*Fy*Sy",
        )
    ]

    if flange_class is FlexuralSlenderness.COMPACT:
        return states

    if flange_class is FlexuralSlenderness.NONCOMPACT:
        Mn = linear_transition(Mp, 0.7 * Fy * Sy, lam, lam_pf, lam_rf)  # Eq. F6-2
        citation, note = cite("F6-2"), "noncompact flange"
    else:
        Fcr = 0.69 * E / lam**2  # Eq. F6-4
        Mn = Fcr * Sy  # Eq. F6-3
        citation, note = cite("F6-3"), f"slender flange, Fcr = {Fcr:.2f} ksi (Eq. F6-4)"

    states.append(
        LimitStateResult(
            LimitState.FLANGE_LOCAL_BUCKLING, Mn, citation,
            detail={"lambda": lam, "lambda_pf": lam_pf, "lambda_rf": lam_rf}, note=note,
        )
    )
    return states


# ===========================================================================
# Sect. F7 -- Square and Rectangular HSS and Box Sections
# ===========================================================================
def hss_effective_flange_width(b: Inch, t: Inch, E: Ksi, Fy: Ksi, *, box: bool = False) -> Inch:
    """Effective width of a slender compression flange.

    AISC 360-16, Eqs. F7-4 (HSS) and F7-5 (box sections), Sect. F7.2, p. 16.1-57::

        be = 1.92*t*sqrt(E/Fy) * (1 - C/(b/t)*sqrt(E/Fy))  <= b

    with ``C = 0.38`` for HSS and ``0.34`` for box sections. The two differ only
    in that coefficient, which is why one function serves both.

    The ``<= b`` cap is active just above ``lambda_r``: at ``b/t = 33.7``
    (Fy = 50) the formula returns essentially ``b`` exactly, so the cap is what
    makes the slender branch continuous with the noncompact one.

    Raises
    ------
    OutOfScopeError
        If the bracket goes non-positive, which happens below
        ``b/t = C*sqrt(E/Fy)`` -- around 9.2 at Fy = 50, far below the
        ``lambda_r`` of 33.7. Reaching that means the flange is not slender and
        Eq. F7-4 does not apply; returning zero effective width instead would be
        absurd for a stocky flange and would silently zero the strength.
    """
    if b <= 0.0 or t <= 0.0:
        raise GeometryError(f"b and t must be positive, got {b}, {t}")
    coefficient = 0.34 if box else 0.38
    root = math.sqrt(E / Fy)
    be = 1.92 * t * root * (1.0 - coefficient / (b / t) * root)
    if be <= 0.0:
        raise OutOfScopeError(
            f"Eq. {'F7-5' if box else 'F7-4'} returns a non-positive effective width "
            f"at b/t = {b / t:.2f}. It applies only to slender flanges "
            f"(b/t > {1.40 * root:.1f} at this grade); this element is not slender."
        )
    return min(be, b)


def effective_section_modulus_rect_hss(
    Ag: Inch2, Ix: Inch4, Ht: Inch, t: Inch, b: Inch, be: Inch
) -> Inch3:
    """Effective section modulus ``Se`` for a rectangular HSS with a slender flange.

    AISC 360-16, Sect. F7.2(c), p. 16.1-57 defines ``Se`` as the "effective
    section modulus determined with the effective width, be, of the compression
    flange" but gives no formula. This computes it the standard way: delete the
    ineffective flange strip, relocate the neutral axis, and refer the modulus
    to the extreme compression fibre.

    **Prefer the tabulated value.** AISC *Manual* Part 1 lists ``Se`` for the
    slender HSS it applies to; pass that instead where available. This helper
    exists for sections outside the Manual, and assumes a flat-sided section
    whose corner geometry the caller has already reflected in ``Ag`` and ``Ix``.
    """
    if min(Ag, Ix, Ht, t, b) <= 0.0:
        raise GeometryError("Ag, Ix, Ht, t and b must be positive")
    if be >= b:
        raise GeometryError(f"be ({be}) is not less than b ({b}); the flange is not slender")

    lost_area = (b - be) * t
    if lost_area >= Ag:
        raise GeometryError(
            f"the ineffective flange strip ({lost_area:.4g} in.^2) exceeds the gross "
            f"area ({Ag:.4g} in.^2) -- check that b is the flat width, not the outside"
        )

    # Origin at the gross centroid; the compression flange centroid sits at +yf.
    yf = (Ht - t) / 2.0
    Ae = Ag - lost_area
    shift = -lost_area * yf / Ae  # neutral axis moves away from the compression flange

    I_about_gross_axis = Ix - (lost_area * yf**2 + (b - be) * t**3 / 12.0)
    I_effective = I_about_gross_axis - Ae * shift**2
    if I_effective <= 0.0:
        raise GeometryError("effective second moment came out non-positive")

    c_compression = Ht / 2.0 - shift
    return I_effective / c_compression


def Lp_F7(ry: Inch, E: Ksi, J: Inch4, Ag: Inch2, Mp: KipIn) -> Inch:
    """Limiting unbraced length for yielding, rectangular HSS.

    AISC 360-16, Eq. F7-12, Sect. F7.4, p. 16.1-58::

        Lp = 0.13*E*ry*sqrt(J*Ag)/Mp

    Unlike Eq. F2-5 this is not a pure geometry ratio -- ``Mp`` appears in the
    denominator, so ``Lp`` depends on the yield stress through the moment.
    """
    if min(ry, Ag) <= 0.0 or Mp <= 0.0:
        raise GeometryError(f"ry, Ag and Mp must be positive, got {ry}, {Ag}, {Mp}")
    if J <= 0.0:
        raise GeometryError(f"J must be positive, got {J}")
    return 0.13 * E * ry * math.sqrt(J * Ag) / Mp


def Lr_F7(ry: Inch, E: Ksi, Fy: Ksi, J: Inch4, Ag: Inch2, Sx: Inch3) -> Inch:
    """Limiting unbraced length for inelastic LTB, rectangular HSS.

    AISC 360-16, Eq. F7-13, Sect. F7.4, p. 16.1-59::

        Lr = 2*E*ry*sqrt(J*Ag)/(0.7*Fy*Sx)
    """
    if min(ry, Ag, Sx) <= 0.0 or J <= 0.0:
        raise GeometryError(f"ry, Ag, Sx and J must be positive, got {ry}, {Ag}, {Sx}, {J}")
    return 2.0 * E * ry * math.sqrt(J * Ag) / (0.7 * Fy * Sx)


def f7_strength(
    member: FlexuralMember,
    *,
    flange_class: FlexuralSlenderness,
    web_class: FlexuralSlenderness,
    b_over_t: Ratio,
    h_over_tw: Ratio,
    b: Inch,
    t: Inch,
    Se: Inch3 | None = None,
) -> list[LimitStateResult]:
    """Sect. F7 -- square and rectangular HSS and box sections.

    AISC 360-16, Sect. F7, pp. 16.1-57 to 16.1-59.

    Note that Eqs. F7-2 and F7-6 do **not** interpolate on lambda between
    lambda_p and lambda_r the way the I-shape sections do. They interpolate
    directly on ``(b/t)*sqrt(Fy/E)`` with hard-coded coefficients -- so
    :func:`pyaisc360.utils.linear_transition` is deliberately not used here.
    """
    E, Fy = member.steel.E, member.steel.Fy
    box = member.shape is ShapeType.BOX
    props = require_properties(member.section, "Ag", "Ix", "ry", "J")
    Ag, Ix, ry, J = props["Ag"], props["Ix"], props["ry"], props["J"]

    minor = member.axis is Axis.MINOR
    Z = require_properties(member.section, "Zy" if minor else "Zx")["Zy" if minor else "Zx"]
    S = require_properties(member.section, "Sy" if minor else "Sx")["Sy" if minor else "Sx"]

    Mp = plastic_moment(Fy, Z)  # Eq. F7-1
    states = [
        LimitStateResult(
            LimitState.PLASTIC_MOMENT, Mp, cite("F7-1"), detail={"Z": Z}, note="Mp = Fy*Z"
        )
    ]

    # -- Flange local buckling, Sect. F7.2 ---------------------------------
    if flange_class is FlexuralSlenderness.NONCOMPACT:
        Mn = min(Mp - (Mp - Fy * S) * (3.57 * b_over_t * math.sqrt(Fy / E) - 4.0), Mp)  # F7-2
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Mn, cite("F7-2"),
                detail={"b/t": b_over_t}, note="noncompact flange",
            )
        )
    elif flange_class is FlexuralSlenderness.SLENDER:
        be = hss_effective_flange_width(b, t, E, Fy, box=box)  # Eq. F7-4 / F7-5
        Se_used = Se if Se is not None else effective_section_modulus_rect_hss(
            Ag, Ix, require_properties(member.section, "Ht")["Ht"], t, b, be
        )
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Fy * Se_used, cite("F7-3"),
                detail={"be": be, "Se": Se_used},
                note="slender flange; Se " + ("supplied" if Se is not None else "computed"),
            )
        )

    # -- Web local buckling, Sect. F7.3 ------------------------------------
    if web_class is FlexuralSlenderness.NONCOMPACT:
        Mn = min(Mp - (Mp - Fy * S) * (0.305 * h_over_tw * math.sqrt(Fy / E) - 0.738), Mp)  # F7-6
        states.append(
            LimitStateResult(
                LimitState.WEB_LOCAL_BUCKLING, Mn, cite("F7-6"),
                detail={"h/tw": h_over_tw}, note="noncompact web",
            )
        )
    elif web_class is FlexuralSlenderness.SLENDER:
        # Sect. F7.3(c) with the User Note on p. 16.1-58: "There are no HSS with
        # slender webs." Reachable only for a built-up box section.
        if not box:
            raise OutOfScopeError(
                "Sect. F7.3 User Note, p. 16.1-58: there are no HSS with slender "
                "webs. A slender web here means the section is a built-up box, or "
                "h/tw was computed from the outside depth instead of the flat width."
            )
        # Sect. F7.3(c): Rpg from Eq. F5-6 with aw = 2*h*tw/(b*tf).
        aw = 2.0 * (h_over_tw * t) * t / (b * t)
        Rpg = Rpg_factor(aw, h_over_tw * t, t, E, Fy)
        states.append(
            LimitStateResult(
                LimitState.COMPRESSION_FLANGE_YIELDING, Rpg * Fy * S, cite("F7-7"),
                detail={"Rpg": Rpg, "aw": aw}, note="slender web, box section",
            )
        )
        # Eq. F7-9 with kc = 4.0 fixed by Sect. F7.3(c), then Eq. F7-8.
        Fcr = 0.9 * E * 4.0 / b_over_t**2
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Rpg * min(Fcr, Fy) * S, cite("F7-8"),
                detail={"Fcr": Fcr, "Rpg": Rpg},
                note=(
                    "slender web, Fcr from Eq. F7-9 with kc = 4.0; the User Note on "
                    "p. 16.1-59 records that Fcr > Fy means another F7 state governs"
                ),
            )
        )

    # -- Lateral-torsional buckling, Sect. F7.4 ----------------------------
    # The User Note on p. 16.1-59 records that LTB cannot occur in a square
    # section or in minor-axis bending; both are excluded rather than computed.
    square = abs(b_over_t - h_over_tw) < 1e-9
    if member.Lb > 0.0 and not minor and not square:
        Lp = Lp_F7(ry, E, J, Ag, Mp)  # Eq. F7-12
        Lr = Lr_F7(ry, E, Fy, J, Ag, S)  # Eq. F7-13
        if member.Lb > Lp:
            if member.Lb <= Lr:
                bracketed = Mp - (Mp - 0.7 * Fy * S) * (member.Lb - Lp) / (Lr - Lp)
                Mn = capped_ltb(member.Cb, bracketed, Mp)  # Eq. F7-10
                citation = cite("F7-10")
                note = "Lp < Lb <= Lr, inelastic LTB"
            else:
                Mn = min(
                    2.0 * E * member.Cb * math.sqrt(J * Ag) / (member.Lb / ry), Mp
                )  # Eq. F7-11
                citation = cite("F7-11")
                note = "Lb > Lr, elastic LTB"
            states.append(
                LimitStateResult(
                    LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, citation,
                    detail={"Lb": member.Lb, "Lp": Lp, "Lr": Lr}, note=note,
                )
            )
    return states


# ===========================================================================
# Sect. F8 -- Round HSS
# ===========================================================================
def f8_strength(
    member: FlexuralMember, *, D_over_t: Ratio, wall_class: FlexuralSlenderness
) -> list[LimitStateResult]:
    """Sect. F8 -- round HSS: yielding and local buckling.

    AISC 360-16, Sect. F8, p. 16.1-59.

    Raises
    ------
    OutOfScopeError
        If ``D/t >= 0.45*E/Fy``. Sect. F8's opening sentence limits the section
        to ratios "less than 0.45E/Fy"; beyond that the Specification gives no
        flexural strength for a round HSS at all.
    """
    E, Fy = member.steel.E, member.steel.Fy
    limit = 0.45 * E / Fy
    if D_over_t >= limit:
        raise OutOfScopeError(
            f"D/t = {D_over_t:.1f} reaches or exceeds 0.45*E/Fy = {limit:.1f}. "
            "AISC 360-16 Sect. F8, p. 16.1-59, does not cover such a round HSS."
        )

    props = require_properties(member.section, "Zx", "Sx")
    Z, S = props["Zx"], props["Sx"]

    states = [
        LimitStateResult(
            LimitState.PLASTIC_MOMENT, plastic_moment(Fy, Z), cite("F8-1"),
            detail={"D/t": D_over_t}, note="Mp = Fy*Z",
        )
    ]

    if wall_class is FlexuralSlenderness.NONCOMPACT:
        Mn = (0.021 * E / D_over_t + Fy) * S  # Eq. F8-2
        states.append(
            LimitStateResult(
                LimitState.LOCAL_BUCKLING, Mn, cite("F8-2"), note="noncompact wall"
            )
        )
    elif wall_class is FlexuralSlenderness.SLENDER:
        Fcr = 0.33 * E / D_over_t  # Eq. F8-4
        states.append(
            LimitStateResult(
                LimitState.LOCAL_BUCKLING, Fcr * S, cite("F8-3"),
                detail={"Fcr": Fcr}, note="slender wall",
            )
        )
    return states


# ===========================================================================
# Sect. F9 -- Tees and Double Angles Loaded in the Plane of Symmetry
# ===========================================================================
def Lp_F9(ry: Inch, E: Ksi, Fy: Ksi) -> Inch:
    """AISC 360-16, Eq. F9-8, Sect. F9.2, p. 16.1-61: ``Lp = 1.76*ry*sqrt(E/Fy)``.

    Numerically identical to Eq. F2-5, but the Specification states it
    separately, so it is cited separately.
    """
    return Lp_F2(ry, E, Fy)


def Lr_F9(ry: Inch, E: Ksi, Fy: Ksi, Iy: Inch4, J: Inch4, Sx: Inch3, d: Inch) -> Inch:
    """Limiting unbraced length for tees and double angles.

    AISC 360-16, Eq. F9-9, Sect. F9.2, p. 16.1-61::

        Lr = 1.95*(E/Fy)*sqrt(Iy*J)/Sx
             * sqrt( 2.36*(Fy/E)*(d*Sx/J) + 1 )
    """
    if min(ry, Iy, Sx, d) <= 0.0 or J <= 0.0:
        raise GeometryError(f"Iy, J, Sx and d must be positive, got {Iy}, {J}, {Sx}, {d}")
    return (
        1.95 * (E / Fy) * math.sqrt(Iy * J) / Sx
        * math.sqrt(2.36 * (Fy / E) * (d * Sx / J) + 1.0)
    )


def Mcr_F9(
    Cb: float, Lb: Inch, E: Ksi, Iy: Inch4, J: Inch4, d: Inch, *, stem_in_compression: bool
) -> KipIn:
    """Elastic LTB moment for a tee or double angle.

    AISC 360-16, Eqs. F9-10, F9-11 and F9-12, Sect. F9.2, p. 16.1-61::

        Mcr = (1.95*E/Lb) * sqrt(Iy*J) * (B + sqrt(1 + B^2))     (Eq. F9-10)
        B   = +2.3*(d/Lb)*sqrt(Iy/J)   stem in tension            (Eq. F9-11)
        B   = -2.3*(d/Lb)*sqrt(Iy/J)   stem in compression        (Eq. F9-12)

    The **sign of B is the whole limit state**. A tee with its stem in
    compression is far weaker than the same tee with the stem in tension,
    because ``B + sqrt(1+B^2)`` collapses towards zero as ``B`` goes negative.
    Getting the sign wrong is unconservative by a large factor, not a small one.
    """
    if Lb <= 0.0:
        raise GeometryError(f"Lb must be positive, got {Lb}")
    if Iy <= 0.0 or J <= 0.0 or d <= 0.0:
        raise GeometryError(f"Iy, J and d must be positive, got {Iy}, {J}, {d}")

    magnitude = 2.3 * (d / Lb) * math.sqrt(Iy / J)
    B = -magnitude if stem_in_compression else magnitude
    return Cb * 1.95 * E / Lb * math.sqrt(Iy * J) * (B + math.sqrt(1.0 + B**2))


def f9_strength(
    member: FlexuralMember,
    *,
    d: Inch,
    Sxc: Inch3,
    flange_class: FlexuralSlenderness,
    lam_f: Ratio,
    lam_pf: Ratio,
    lam_rf: Ratio,
    d_over_tw: Ratio,
) -> list[LimitStateResult]:
    """Sect. F9 -- tees and double angles loaded in the plane of symmetry.

    AISC 360-16, Sect. F9, pp. 16.1-60 to 16.1-62.

    Sect. F9.1 states the yielding limit state as ``Mn = Mp`` (Eq. F9-1); which
    of Eqs. F9-2, F9-4 and F9-5 defines ``Mp`` depends on whether the stem or
    web leg is in tension or compression, and on tee versus double angle.
    """
    E, Fy = member.steel.E, member.steel.Fy
    props = require_properties(member.section, "Zx", "Sx", "Iy", "J", "ry")
    Zx, Sx, Iy, J, ry = props["Zx"], props["Sx"], props["Iy"], props["J"], props["ry"]

    My = yield_moment(Fy, Sx)  # Eq. F9-3
    double_angle = member.shape is ShapeType.DOUBLE_ANGLE

    # -- Yielding, Sect. F9.1 ----------------------------------------------
    if member.stem_in_compression:
        if double_angle:
            Mp = 1.5 * My  # Eq. F9-5, double angles with web legs in compression
            yield_citation = cite("F9-5")
        else:
            Mp = My  # Eq. F9-4, tee stems in compression
            yield_citation = cite("F9-4")
    else:
        Mp = min(Fy * Zx, 1.6 * My)  # Eq. F9-2, stems and web legs in tension
        yield_citation = cite("F9-2")

    states = [
        LimitStateResult(
            LimitState.PLASTIC_MOMENT, Mp, yield_citation,
            detail={"My": My, "Zx": Zx},
            note="stem in compression" if member.stem_in_compression else "stem in tension",
        )
    ]

    # -- Lateral-torsional buckling, Sect. F9.2 ----------------------------
    if member.Lb > 0.0:
        if member.stem_in_compression:
            # Sect. F9.2(b): Mcr from Eq. F9-10 with B negative, capped at My.
            Mcr = Mcr_F9(member.Cb, member.Lb, E, Iy, J, d, stem_in_compression=True)
            Mn = min(Mcr, My)  # Eq. F9-13
            states.append(
                LimitStateResult(
                    LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, cite("F9-13"),
                    detail={"Mcr": Mcr, "My": My}, note="stem in compression, Mn = Mcr <= My",
                )
            )
        else:
            Lp = Lp_F9(ry, E, Fy)  # Eq. F9-8
            Lr = Lr_F9(ry, E, Fy, Iy, J, Sx, d)  # Eq. F9-9
            if member.Lb > Lp:
                if member.Lb <= Lr:
                    Mn = Mp - (Mp - My) * (member.Lb - Lp) / (Lr - Lp)  # Eq. F9-6
                    citation, note = cite("F9-6"), "Lp < Lb <= Lr"
                else:
                    Mn = Mcr_F9(
                        member.Cb, member.Lb, E, Iy, J, d, stem_in_compression=False
                    )  # Eqs. F9-7 / F9-10
                    citation, note = cite("F9-7"), "Lb > Lr, Mn = Mcr"
                states.append(
                    LimitStateResult(
                        LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, citation,
                        detail={"Lb": member.Lb, "Lp": Lp, "Lr": Lr}, note=note,
                    )
                )

    # -- Flange local buckling, Sect. F9.3 ---------------------------------
    if flange_class is FlexuralSlenderness.NONCOMPACT:
        Mn = min(
            linear_transition(Mp, 0.7 * Fy * Sxc, lam_f, lam_pf, lam_rf), 1.6 * My
        )  # Eq. F9-14
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Mn, cite("F9-14"), note="noncompact flange"
            )
        )
    elif flange_class is FlexuralSlenderness.SLENDER:
        Mn = 0.7 * E * Sxc / lam_f**2  # Eq. F9-15
        states.append(
            LimitStateResult(
                LimitState.FLANGE_LOCAL_BUCKLING, Mn, cite("F9-15"), note="slender flange"
            )
        )

    # -- Local buckling of tee stems, Sect. F9.4 ---------------------------
    if member.stem_in_compression and not double_angle:
        root = math.sqrt(E / Fy)
        if d_over_tw <= 0.84 * root:
            Fcr = Fy  # Eq. F9-17
            citation = cite("F9-17")
        elif d_over_tw <= 1.52 * root:
            Fcr = (1.43 - 0.515 * d_over_tw * math.sqrt(Fy / E)) * Fy  # Eq. F9-18
            citation = cite("F9-18")
        else:
            Fcr = 1.52 * E / d_over_tw**2  # Eq. F9-19
            citation = cite("F9-19")
        states.append(
            LimitStateResult(
                LimitState.WEB_LOCAL_BUCKLING, Fcr * Sx, cite("F9-16"),
                detail={"Fcr": Fcr, "d/tw": d_over_tw},
                note=f"tee stem in compression, Fcr from {citation.equation}",
            )
        )
    return states


# ===========================================================================
# Sect. F10 -- Single Angles
# ===========================================================================
def Mcr_F10_principal(
    Cb: float, E: Ksi, Ag: Inch2, rz: Inch, t: Inch, Lb: Inch, beta_w: Inch
) -> KipIn:
    """Elastic LTB moment about the major principal axis of a single angle.

    AISC 360-16, Eq. F10-4, Sect. F10.2, p. 16.1-63::

        Mcr = (9*E*A*rz*t*Cb/(8*Lb)) * [ sqrt(1 + (4.4*beta_w*rz/(Lb*t))^2)
                                         + 4.4*beta_w*rz/(Lb*t) ]

    ``beta_w`` is **signed**: positive with short legs in compression, negative
    with long legs in compression, zero for an equal-leg angle. Sect. F10.2
    requires the negative value wherever the long leg is in compression anywhere
    along the unbraced length -- so a member that reverses curvature takes the
    negative value throughout.

    ``Cb`` here is capped at 1.5 (Sect. F10.2), the only such cap in Chapter F.
    """
    if Lb <= 0.0 or Ag <= 0.0 or rz <= 0.0 or t <= 0.0:
        raise GeometryError(f"Lb, Ag, rz and t must be positive, got {Lb}, {Ag}, {rz}, {t}")
    Cb_capped = min(Cb, CB_ANGLE_MAX)
    term = 4.4 * beta_w * rz / (Lb * t)
    return (9.0 * E * Ag * rz * t * Cb_capped / (8.0 * Lb)) * (
        math.sqrt(1.0 + term**2) + term
    )


def Mcr_F10_geometric(
    Cb: float, E: Ksi, b: Inch, t: Inch, Lb: Inch, *, compression_at_toe: bool,
    restrained_at_max_moment: bool = False,
) -> KipIn:
    """Elastic LTB moment about a geometric axis of an **equal-leg** angle.

    AISC 360-16, Eqs. F10-5a and F10-5b, Sect. F10.2(2)(i), p. 16.1-64::

        Mcr = (0.58*E*b^4*t*Cb/Lb^2) * [ sqrt(1 + 0.88*(Lb*t/b^2)^2) -/+ 1 ]

    with the minus sign for maximum compression at the toe (Eq. F10-5a) and the
    plus sign for maximum tension at the toe (Eq. F10-5b). Compression at the
    toe is much the weaker case.

    Parameters
    ----------
    restrained_at_max_moment:
        Sect. F10.2(2)(ii): with lateral-torsional restraint at the point of
        maximum moment only, ``Mcr`` is taken as 1.25 times the value from
        Eq. F10-5a or F10-5b, and ``My`` reverts to the full geometric-axis
        yield moment rather than 0.80 of it.
    """
    if Lb <= 0.0 or b <= 0.0 or t <= 0.0:
        raise GeometryError(f"Lb, b and t must be positive, got {Lb}, {b}, {t}")
    Cb_capped = min(Cb, CB_ANGLE_MAX)
    inner = math.sqrt(1.0 + 0.88 * (Lb * t / b**2) ** 2)
    Mcr = (0.58 * E * b**4 * t * Cb_capped / Lb**2) * (
        inner - 1.0 if compression_at_toe else inner + 1.0
    )
    return 1.25 * Mcr if restrained_at_max_moment else Mcr


def f10_strength(
    member: FlexuralMember,
    *,
    My: KipIn,
    Sc: Inch3,
    b_over_t: Ratio,
    leg_class: FlexuralSlenderness,
    Mcr: KipIn | None = None,
    toe_in_compression: bool = True,
) -> list[LimitStateResult]:
    """Sect. F10 -- single angles: yielding, LTB and leg local buckling.

    AISC 360-16, Sect. F10, pp. 16.1-62 to 16.1-65.

    Parameters
    ----------
    My:
        Yield moment about the axis of bending. Sect. F10.2(2)(i) requires this
        to be **0.80 times** the moment from the geometric section modulus for
        an equal-leg angle bent about a geometric axis with no lateral-torsional
        restraint -- the caller applies that factor.
    Sc:
        Elastic section modulus to the toe in compression. Same 0.80 factor
        applies for the unrestrained geometric-axis case.
    Mcr:
        Elastic LTB moment from Eq. F10-4 or F10-5; omit when the angle has
        continuous lateral-torsional restraint, in which case Sect. F10.2 does
        not apply.
    """
    E, Fy = member.steel.E, member.steel.Fy

    states = [
        LimitStateResult(
            LimitState.YIELDING, 1.5 * My, cite("F10-1"),
            detail={"My": My}, note="Mn = 1.5*My",
        )
    ]

    # -- Lateral-torsional buckling, Sect. F10.2 ---------------------------
    if Mcr is not None and not member.continuous_restraint:
        if My / Mcr <= 1.0:
            Mn = min((1.92 - 1.17 * math.sqrt(My / Mcr)) * My, 1.5 * My)  # Eq. F10-2
            citation, note = cite("F10-2"), "My/Mcr <= 1.0"
        else:
            Mn = (0.92 - 0.17 * Mcr / My) * Mcr  # Eq. F10-3
            citation, note = cite("F10-3"), "My/Mcr > 1.0"
        states.append(
            LimitStateResult(
                LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, citation,
                detail={"Mcr": Mcr, "My": My, "My/Mcr": My / Mcr}, note=note,
            )
        )

    # -- Leg local buckling, Sect. F10.3 -----------------------------------
    # "applies when the toe of the leg is in compression" -- otherwise skipped.
    if toe_in_compression and leg_class is not FlexuralSlenderness.COMPACT:
        if leg_class is FlexuralSlenderness.NONCOMPACT:
            Mn = Fy * Sc * (2.43 - 1.72 * b_over_t * math.sqrt(Fy / E))  # Eq. F10-6
            citation, note = cite("F10-6"), "noncompact leg"
        else:
            Fcr = 0.71 * E / b_over_t**2  # Eq. F10-8
            Mn = Fcr * Sc  # Eq. F10-7
            citation, note = cite("F10-7"), f"slender leg, Fcr = {Fcr:.2f} ksi"
        states.append(
            LimitStateResult(
                LimitState.LEG_LOCAL_BUCKLING, Mn, citation,
                detail={"b/t": b_over_t, "Sc": Sc}, note=note,
            )
        )
    return states


# ===========================================================================
# Sect. F11 -- Rectangular Bars and Rounds
# ===========================================================================
def f11_strength(
    member: FlexuralMember, *, d: Inch, t: Inch, Z: Inch3, S: Inch3
) -> list[LimitStateResult]:
    """Sect. F11 -- rectangular bars and rounds.

    AISC 360-16, Sect. F11, pp. 16.1-65 to 16.1-66.

    The slenderness parameter here is ``Lb*d/t^2``, not ``Lb/r`` -- a solid
    rectangle has no meaningful warping constant, so its LTB is governed by
    pure torsion and the aspect ratio.

    Parameters
    ----------
    d:
        Depth of the rectangular bar, in.
    t:
        Width of the bar **parallel to the axis of bending**, in.
    """
    E, Fy = member.steel.E, member.steel.Fy
    Mp = plastic_moment(Fy, Z, S=S)  # Eq. F11-1, capped at 1.6*Fy*S
    My = yield_moment(Fy, S)

    round_or_minor = member.shape is ShapeType.ROUND_BAR or member.axis is Axis.MINOR
    states = [
        LimitStateResult(
            LimitState.PLASTIC_MOMENT, Mp, cite("F11-1"),
            detail={"Z": Z, "S": S}, note="Mp = Fy*Z <= 1.6*Fy*S",
        )
    ]

    # Sect. F11.2(d): LTB need not be considered for rounds or minor-axis bending.
    if round_or_minor or member.Lb <= 0.0:
        return states

    if min(d, t) <= 0.0:
        raise GeometryError(f"d and t must be positive, got {d}, {t}")
    parameter = member.Lb * d / t**2
    lower = 0.08 * E / Fy
    upper = 1.9 * E / Fy

    if parameter <= lower:
        # Sect. F11.2(a): LTB does not apply.
        return states

    if parameter <= upper:
        bracketed = (1.52 - 0.274 * parameter * Fy / E) * My
        Mn = capped_ltb(member.Cb, bracketed, Mp)  # Eq. F11-2
        citation, note = cite("F11-2"), "0.08E/Fy < Lb*d/t^2 <= 1.9E/Fy"
    else:
        Fcr = 1.9 * E * member.Cb / parameter  # Eq. F11-4
        Mn = min(Fcr * S, Mp)  # Eq. F11-3
        citation, note = cite("F11-3"), f"Lb*d/t^2 > 1.9E/Fy, Fcr = {Fcr:.2f} ksi"

    states.append(
        LimitStateResult(
            LimitState.LATERAL_TORSIONAL_BUCKLING, Mn, citation,
            detail={"Lb*d/t^2": parameter, "0.08E/Fy": lower, "1.9E/Fy": upper}, note=note,
        )
    )
    return states


# ===========================================================================
# Sect. F12 -- Unsymmetrical Shapes
# ===========================================================================
def f12_strength(
    member: FlexuralMember,
    *,
    S_min: Inch3,
    Fcr_ltb: Ksi | None = None,
    Fcr_local: Ksi | None = None,
) -> list[LimitStateResult]:
    """Sect. F12 -- unsymmetrical shapes other than single angles.

    AISC 360-16, Sect. F12, pp. 16.1-66 to 16.1-67::

        Mn = Fn*S_min                                       (Eq. F12-1)
        yielding:        Fn = Fy                            (Eq. F12-2)
        LTB:             Fn = Fcr <= Fy                     (Eq. F12-3)
        local buckling:  Fn = Fcr <= Fy                     (Eq. F12-4)

    Both buckling stresses are "as determined by analysis" -- the Specification
    supplies no formula, so they are inputs. Omit them and only yielding is
    evaluated, with a note saying so; that is a genuinely incomplete check and
    the report says as much rather than implying a full one.

    The User Note on p. 16.1-66 records that these provisions can be overly
    conservative and points to Appendix 1.3 as an alternative.
    """
    Fy = member.steel.Fy
    states = [
        LimitStateResult(
            LimitState.YIELDING, Fy * S_min, cite("F12-2"),
            detail={"S_min": S_min}, note="Mn = Fy*S_min per Eqs. F12-1, F12-2",
        )
    ]
    if Fcr_ltb is not None:
        states.append(
            LimitStateResult(
                LimitState.LATERAL_TORSIONAL_BUCKLING, min(Fcr_ltb, Fy) * S_min,
                cite("F12-3"), note="Fcr from analysis, per Sect. F12.2",
            )
        )
    if Fcr_local is not None:
        states.append(
            LimitStateResult(
                LimitState.LOCAL_BUCKLING, min(Fcr_local, Fy) * S_min,
                cite("F12-4"), note="Fcr from analysis, per Sect. F12.3",
            )
        )
    if Fcr_ltb is None or Fcr_local is None:
        states[0] = LimitStateResult(
            states[0].limit_state, states[0].nominal, states[0].citation,
            detail=states[0].detail,
            note=(
                "INCOMPLETE: Sect. F12 also requires lateral-torsional and local "
                "buckling stresses determined by analysis; supply Fcr_ltb / Fcr_local"
            ),
        )
    return states


# ===========================================================================
# Sect. F13 -- Proportions of Beams and Girders
# ===========================================================================
def tension_rupture_cap(Fu: Ksi, Afn: Inch2, Afg: Inch2, Fy: Ksi, Sx: Inch3) -> KipIn | None:
    """Flexural strength cap from tensile rupture of a bolted tension flange.

    AISC 360-16, Sect. F13.1, Eq. F13-1, p. 16.1-67::

        Fu*Afn >= Yt*Fy*Afg:  the limit state does not apply
        Fu*Afn <  Yt*Fy*Afg:  Mn <= (Fu*Afn/Afg)*Sx
        Yt = 1.0 for Fy/Fu <= 0.8, otherwise 1.1

    Returns ``None`` when the limit state does not apply. This is a *cap* on
    whatever Sects. F2-F12 produced at the location of the holes, not an
    independent limit state -- so it is applied by the caller rather than added
    to the limit-state list.
    """
    if min(Afn, Afg, Sx) <= 0.0:
        raise GeometryError(f"Afn, Afg and Sx must be positive, got {Afn}, {Afg}, {Sx}")
    if Afn > Afg:
        raise GeometryError(f"net area {Afn} exceeds gross area {Afg}")

    Yt = 1.0 if Fy / Fu <= 0.8 else 1.1
    if Fu * Afn >= Yt * Fy * Afg:
        return None
    return (Fu * Afn / Afg) * Sx  # Eq. F13-1


def check_proportioning_limits(Iyc: Inch4, Iy: Inch4) -> None:
    """Singly symmetric I-shape proportioning limit.

    AISC 360-16, Eq. F13-2, Sect. F13.2, p. 16.1-67::

        0.1 <= Iyc/Iy <= 0.9

    Raises
    ------
    OutOfScopeError
        Outside that band. The Chapter F equations are calibrated on tests
        within it; a section outside is not a Chapter F member at all.
    """
    if Iy <= 0.0:
        raise GeometryError(f"Iy must be positive, got {Iy}")
    ratio = Iyc / Iy
    if not 0.1 <= ratio <= 0.9:
        raise OutOfScopeError(
            f"Eq. F13-2, p. 16.1-67: Iyc/Iy = {ratio:.4g} is outside the required "
            "band 0.1 <= Iyc/Iy <= 0.9 for a singly symmetric I-shaped member"
        )


def cover_plate_extension(w: Inch, *, end_weld: str = "none") -> Inch:
    """Length ``a'`` a partial-length cover plate must extend past its cutoff.

    AISC 360-16, Eqs. F13-5, F13-6 and F13-7, Sect. F13.3(e), p. 16.1-69::

        continuous end weld >= 3/4 of the plate thickness:  a' = w    (F13-5)
        continuous end weld <  3/4 of the plate thickness:  a' = 1.5w (F13-6)
        no weld across the end of the plate:                a' = 2w   (F13-7)

    Parameters
    ----------
    w:
        Width of the cover plate, in.
    end_weld:
        ``"thick"``, ``"thin"`` or ``"none"`` for the three cases above.
    """
    if w <= 0.0:
        raise GeometryError(f"cover plate width must be positive, got {w}")
    factors = {"thick": 1.0, "thin": 1.5, "none": 2.0}
    if end_weld not in factors:
        raise AISC360Error(f"end_weld must be one of {sorted(factors)}, got {end_weld!r}")
    return factors[end_weld] * w


def Lm_moment_redistribution(
    M1_over_M2: Ratio, ry: Inch, E: Ksi, Fy: Ksi, *, solid_or_box: bool = False
) -> Inch:
    """Maximum unbraced length ``Lm`` permitting moment redistribution.

    AISC 360-16, Eqs. F13-8 and F13-9, Sect. F13.5, p. 16.1-69::

        I-shapes:            Lm = [0.12 + 0.076*(M1/M2)]*(E/Fy)*ry      (F13-8)
        bars and box beams:  Lm = [0.17 + 0.10*(M1/M2)]*(E/Fy)*ry
                                  >= 0.10*(E/Fy)*ry                     (F13-9)

    ``M1/M2`` is **positive when the moments cause reverse curvature** and
    negative for single curvature -- the same convention Eq. A-8-4 uses for
    ``Cm``, so the two can share a sign without conversion.

    Sect. B3.3 permits the 0.9 redistribution of negative end moments only if
    the compression flange is braced within this length.

    Notes
    -----
    Sect. F13.5 places no limit on ``Lb`` for round or square cross-sections, or
    for any beam bent about its minor axis; those cases should not call this.
    """
    if ry <= 0.0:
        raise GeometryError(f"ry must be positive, got {ry}")
    if not -1.0 <= M1_over_M2 <= 1.0:
        raise GeometryError(
            f"M1/M2 = {M1_over_M2} is outside [-1, 1]; M1 is the *smaller* end moment"
        )
    base = (E / Fy) * ry
    if solid_or_box:
        return max((0.17 + 0.10 * M1_over_M2) * base, 0.10 * base)  # Eq. F13-9
    return (0.12 + 0.076 * M1_over_M2) * base  # Eq. F13-8


def web_slenderness_limit(E: Ksi, Fy: Ksi, a_over_h: Ratio | None) -> Ratio:
    """Maximum ``h/tw`` for an I-shaped member with a slender web.

    AISC 360-16, Eqs. F13-3 and F13-4, Sect. F13.2, p. 16.1-68::

        a/h <= 1.5:  (h/tw)max = 12.0*sqrt(E/Fy)          (Eq. F13-3)
        a/h >  1.5:  (h/tw)max = 0.40*E/Fy                (Eq. F13-4)

    Parameters
    ----------
    a_over_h:
        Ratio of clear stiffener spacing to web depth. ``None`` for an
        unstiffened girder, where Sect. F13.2 instead caps ``h/tw`` at 260.
    """
    if a_over_h is None:
        return 260.0
    if a_over_h <= 1.5:
        return 12.0 * math.sqrt(E / Fy)  # Eq. F13-3
    return 0.40 * E / Fy  # Eq. F13-4


# ===========================================================================
# Orchestrator
# ===========================================================================
def flexural_strength(
    member: FlexuralMember,
    *,
    basis: Basis = Basis.LRFD,
    settings: DesignSettings = DEFAULT_SETTINGS,
    **section_kwargs: Any,
) -> StrengthResult:
    """Nominal and available flexural strength, ``Mn``, in kip-in.

    AISC 360-16, Sect. F1, p. 16.1-46: ``phi_b = 0.90`` (LRFD),
    ``Omega_b = 1.67`` (ASD), for every provision in the chapter.

    Dispatches to the applicable section by shape family, bending axis, and --
    for I-shapes and channels -- the flange and web classifications, which
    select between Sects. F2, F3, F4 and F5:

    ==================  ==============  =======
    Web                 Flange          Section
    ==================  ==============  =======
    compact             compact         F2
    compact             non/slender     F3
    noncompact          any             F4
    slender             any             F5
    ==================  ==============  =======

    Extra keyword arguments are forwarded to the selected section function; see
    those for what each needs. This is the one place the library takes ``**kwargs``,
    because the sections genuinely require different inputs and inventing a
    single union type for all of them would hide that.
    """
    shape = member.shape
    minor = member.axis is Axis.MINOR

    if shape in {ShapeType.ROLLED_I, ShapeType.BUILT_UP_I, ShapeType.CHANNEL}:
        if minor:
            states = f6_strength(member, **section_kwargs)
        else:
            states = _dispatch_i_shape(member, **section_kwargs)
    elif shape in {ShapeType.RECTANGULAR_HSS, ShapeType.BOX}:
        states = f7_strength(member, **section_kwargs)
    elif shape is ShapeType.ROUND_HSS:
        states = f8_strength(member, **section_kwargs)
    elif shape in {ShapeType.TEE, ShapeType.DOUBLE_ANGLE}:
        states = f9_strength(member, **section_kwargs)
    elif shape is ShapeType.SINGLE_ANGLE:
        states = f10_strength(member, **section_kwargs)
    elif shape in {ShapeType.RECTANGULAR_BAR, ShapeType.ROUND_BAR}:
        states = f11_strength(member, **section_kwargs)
    elif shape is ShapeType.UNSYMMETRICAL:
        states = f12_strength(member, **section_kwargs)
    else:  # pragma: no cover - the enum is exhaustive
        raise AISC360Error(f"no Chapter F section covers {shape}")

    return StrengthResult.build(
        "Mn", "kip-in.", states, phi=PHI_B, omega=OMEGA_B, basis=basis
    )


def _dispatch_i_shape(
    member: FlexuralMember,
    *,
    flange_class: FlexuralSlenderness,
    web_class: FlexuralSlenderness,
    **kwargs: Any,
) -> list[LimitStateResult]:
    """Select among Sects. F2-F5 from the flange and web classifications."""
    if web_class is FlexuralSlenderness.SLENDER:
        return f5_strength(member, flange_class=flange_class, **kwargs)
    if web_class is FlexuralSlenderness.NONCOMPACT:
        return f4_strength(member, flange_class=flange_class, **kwargs)
    # Compact web.
    if flange_class is FlexuralSlenderness.COMPACT:
        return f2_strength(member, **{k: v for k, v in kwargs.items() if k == "c"})
    return f3_strength(member, flange_class=flange_class, **kwargs)


def selected_section(
    flange_class: FlexuralSlenderness, web_class: FlexuralSlenderness
) -> str:
    """Which of Sects. F2-F5 governs an I-shape, as a label for reports."""
    if web_class is FlexuralSlenderness.SLENDER:
        return "F5"
    if web_class is FlexuralSlenderness.NONCOMPACT:
        return "F4"
    return "F2" if flange_class is FlexuralSlenderness.COMPACT else "F3"


def limit_states_of(states: Sequence[LimitStateResult]) -> set[LimitState]:
    """Set of limit states present -- convenience for tests and reports."""
    return {s.limit_state for s in states}
