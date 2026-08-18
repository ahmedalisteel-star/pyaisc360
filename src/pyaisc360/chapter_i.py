"""Chapter I -- Design of Composite Members.

Covers Sects. I1 through I8, pp. 16.1-86 to 16.1-112.

Chapter I is the only chapter that designs **two materials at once**, and that
shows up in three ways the steel-only chapters never have to deal with:

**Its compression factor is different.** Sect. I2 uses ``phi_c = 0.75`` /
``Omega_c = 2.00``, not Chapter E's 0.90 / 1.67. The extra margin covers
concrete's variability, and using Chapter E's factor on a composite column
overstates it by 20%.

**Material limits are mandatory, not advisory.** Sect. I1.3 caps ``f'c`` at 10
ksi (6 for lightweight), ``Fy`` at 75 ksi and ``Fysr`` at 80 ksi *for strength
calculations*. The User Note is explicit that higher-strength concrete "may be
used for stiffness calculations but may not be relied upon for strength" -- so
the limit applies asymmetrically, and :func:`check_material_limits` says which
use is being made.

**The buckling equations look like Chapter E's but are not.** Eqs. I2-2 and
I2-3 mirror Eqs. E3-2 and E3-3 in shape, but switch on ``Pno/Pe`` rather than
``Fy/Fe``, and ``Pno`` is a *squash load* built from three materials rather than
a stress. The 2.25 threshold and the 0.658 and 0.877 coefficients are the same
numbers doing the same job on a different quantity.
"""

from __future__ import annotations

import math
from enum import Enum

from .core.citations import cite
from .core.config import Basis
from .core.enums import FlexuralSlenderness, LimitState
from .core.exceptions import AISC360Error, GeometryError, OutOfScopeError
from .core.result import InteractionResult, LimitStateResult, StrengthResult
from .core.units import Inch, Inch2, Inch4, Kip, KipIn, Ksi, Ratio

__all__ = [
    "PHI_C_COMPOSITE",
    "OMEGA_C_COMPOSITE",
    "PHI_T_COMPOSITE",
    "OMEGA_T_COMPOSITE",
    "PHI_B_COMPOSITE",
    "OMEGA_B_COMPOSITE",
    "PHI_BEARING",
    "OMEGA_BEARING",
    "PHI_BOND",
    "OMEGA_BOND",
    "PHI_V_ANCHOR",
    "OMEGA_V_ANCHOR",
    "PHI_T_ANCHOR",
    "OMEGA_T_ANCHOR",
    "FC_MIN",
    "FC_MAX_NORMAL_WEIGHT",
    "FC_MAX_LIGHTWEIGHT",
    "FY_MAX",
    "FYSR_MAX",
    "MIN_REINFORCEMENT_RATIO",
    "CompositeType",
    "DeckOrientation",
    "concrete_modulus",
    "check_material_limits",
    "reinforcement_ratio",
    # I2
    "Pno_encased",
    "C1_coefficient",
    "EIeff_encased",
    "Pe_composite",
    "composite_compressive_strength",
    "composite_tensile_strength",
    "Pp_filled",
    "Py_filled",
    "Pno_filled",
    "Fcr_filled_rectangular",
    "Fcr_filled_round",
    "C3_coefficient",
    "EIeff_filled",
    # I3
    "horizontal_shear_positive",
    "horizontal_shear_negative",
    "filled_flexural_strength",
    # I5
    "csr_ratio",
    "cp_cm_coefficients",
    "composite_interaction",
    # I6
    "force_to_concrete",
    "force_to_steel",
    "direct_bearing_strength",
    "shear_connection_strength",
    "bond_stress",
    "bond_strength",
    # I8
    "Rg_Rp",
    "stud_anchor_shear_in_beam",
    "channel_anchor_shear",
    "stud_shear_strength",
    "stud_tensile_strength",
    "anchor_interaction",
]

#: Sect. I2.1b and I2.2b -- composite compression. **Not** Chapter E's
#: 0.90/1.67: the extra margin covers concrete's variability.
PHI_C_COMPOSITE: float = 0.75
OMEGA_C_COMPOSITE: float = 2.00
#: Sect. I2.1c and I2.2c -- composite tension, on the steel alone.
PHI_T_COMPOSITE: float = 0.90
OMEGA_T_COMPOSITE: float = 1.67
#: Sects. I3.3 and I3.4b -- composite flexure.
PHI_B_COMPOSITE: float = 0.90
OMEGA_B_COMPOSITE: float = 1.67
#: Sect. I6.3a -- direct bearing on concrete. Same pair as Sect. J8.
PHI_BEARING: float = 0.65
OMEGA_BEARING: float = 2.31
#: Sect. I6.3c -- direct bond interaction. The lowest phi in the Specification.
PHI_BOND: float = 0.50
OMEGA_BOND: float = 3.00
#: Sect. I8.3a -- steel headed stud anchor in shear.
PHI_V_ANCHOR: float = 0.65
OMEGA_V_ANCHOR: float = 2.31
#: Sect. I8.3b -- steel headed stud anchor in tension.
PHI_T_ANCHOR: float = 0.75
OMEGA_T_ANCHOR: float = 2.00

#: Sect. I1.3(a), p. 16.1-88 -- concrete compressive strength limits, ksi.
FC_MIN: Ksi = 3.0
FC_MAX_NORMAL_WEIGHT: Ksi = 10.0
FC_MAX_LIGHTWEIGHT: Ksi = 6.0
#: Sect. I1.3(b) and I1.3(c) -- yield stress caps, ksi.
FY_MAX: Ksi = 75.0
FYSR_MAX: Ksi = 80.0

#: Sect. I2.1a(c), Eq. I2-1 -- minimum continuous longitudinal reinforcement.
MIN_REINFORCEMENT_RATIO: Ratio = 0.004


class CompositeType(str, Enum):
    """Which family of composite member. Selects the Sect. I2 branch."""

    ENCASED = "concrete-encased"
    FILLED_RECTANGULAR = "concrete-filled rectangular HSS"
    FILLED_ROUND = "concrete-filled round HSS"


class DeckOrientation(str, Enum):
    """Steel deck orientation relative to the beam, for Sect. I8.2a's Rg/Rp."""

    NONE = "no decking"
    PARALLEL = "deck parallel to the steel shape"
    PERPENDICULAR = "deck perpendicular to the steel shape"


# ===========================================================================
# Sect. I1 -- General provisions
# ===========================================================================
def concrete_modulus(wc: float, fc_prime: Ksi) -> Ksi:
    """Modulus of elasticity of concrete.

    AISC 360-16, Sect. I2.1b, p. 16.1-91::

        Ec = wc^1.5*sqrt(f'c),  ksi, with wc in lb/ft^3

    Normal-weight concrete at ``wc = 145`` pcf and ``f'c = 4`` ksi gives
    ``Ec = 3492`` ksi -- about an eighth of steel's 29,000.

    The Specification restricts ``wc`` to 90-155 lb/ft^3 (1500-2500 kg/m^3).
    """
    if not 90.0 <= wc <= 155.0:
        raise OutOfScopeError(
            f"unit weight {wc} lb/ft^3 is outside the 90-155 range Sect. I2.1b covers"
        )
    if fc_prime <= 0.0:
        raise GeometryError(f"f'c must be positive, got {fc_prime}")
    return math.pow(wc, 1.5) * math.sqrt(fc_prime)


def check_material_limits(
    fc_prime: Ksi, Fy: Ksi, Fysr: Ksi = 0.0, *,
    lightweight: bool = False, for_strength: bool = True,
) -> list[str]:
    """Sect. I1.3 material limitations, p. 16.1-88.

    Returns the list of violations::

        3 ksi <= f'c <= 10 ksi   normal weight
        3 ksi <= f'c <=  6 ksi   lightweight
        Fy   <= 75 ksi
        Fysr <= 80 ksi

    The ``f'c`` cap applies **asymmetrically**. Sect. I1.3(a)'s User Note:
    "Higher strength concrete material properties may be used for **stiffness**
    calculations but may not be relied upon for **strength** calculations unless
    justified by testing or analysis." So ``for_strength=False`` relaxes only the
    upper bound -- the 3 ksi floor still applies, since a weaker concrete is not
    covered either way.

    Reports rather than raising, because a violation makes the *equations*
    inapplicable rather than the member illegal, and the caller may be doing a
    stiffness calculation.
    """
    violations: list[str] = []
    ceiling = FC_MAX_LIGHTWEIGHT if lightweight else FC_MAX_NORMAL_WEIGHT

    if fc_prime < FC_MIN:
        violations.append(f"Sect. I1.3(a): f'c = {fc_prime:g} ksi is below {FC_MIN:g} ksi")
    if for_strength and fc_prime > ceiling:
        kind = "lightweight" if lightweight else "normal weight"
        violations.append(
            f"Sect. I1.3(a): f'c = {fc_prime:g} ksi exceeds {ceiling:g} ksi for {kind} "
            "concrete in a STRENGTH calculation (the User Note permits higher f'c "
            "for stiffness only)"
        )
    if Fy > FY_MAX:
        violations.append(f"Sect. I1.3(b): Fy = {Fy:g} ksi exceeds {FY_MAX:g} ksi")
    if Fysr > FYSR_MAX:
        violations.append(f"Sect. I1.3(c): Fysr = {Fysr:g} ksi exceeds {FYSR_MAX:g} ksi")
    return violations


def reinforcement_ratio(Asr: Inch2, Ag: Inch2) -> tuple[Ratio, bool]:
    """Continuous longitudinal reinforcement ratio for an encased member.

    AISC 360-16, Eq. I2-1, Sect. I2.1a(c), p. 16.1-90::

        rho_sr = Asr/Ag,  minimum 0.004

    Returns ``(rho_sr, meets_minimum)``. The 0.004 floor applies to **encased**
    members only -- Sect. I2.2a(c) states that "minimum longitudinal
    reinforcement is not required" for filled members, because the steel tube
    already confines the concrete.
    """
    if Ag <= 0.0:
        raise GeometryError(f"Ag must be positive, got {Ag}")
    if Asr < 0.0:
        raise GeometryError(f"Asr must be non-negative, got {Asr}")
    rho = Asr / Ag
    return rho, rho >= MIN_REINFORCEMENT_RATIO


# ===========================================================================
# Sect. I2 -- Axial force
# ===========================================================================
def Pno_encased(Fy: Ksi, As: Inch2, Fysr: Ksi, Asr: Inch2, fc_prime: Ksi, Ac: Inch2) -> Kip:
    """Squash load of an encased composite member.

    AISC 360-16, Eq. I2-4, Sect. I2.1b, p. 16.1-91::

        Pno = Fy*As + Fysr*Asr + 0.85*f'c*Ac

    The ``0.85`` on the concrete is the standard rectangular-stress-block factor.
    Note it is **0.85 for encased** members but ``C2`` (0.85 rectangular, **0.95
    round**) for filled ones -- a round tube confines its concrete and gets the
    higher value.
    """
    for name, value in (("Fy", Fy), ("As", As), ("fc_prime", fc_prime), ("Ac", Ac)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    return Fy * As + Fysr * Asr + 0.85 * fc_prime * Ac


def C1_coefficient(As: Inch2, Asr: Inch2, Ag: Inch2) -> Ratio:
    """Effective rigidity coefficient for an **encased** member.

    AISC 360-16, Eq. I2-7, Sect. I2.1b, p. 16.1-92::

        C1 = 0.25 + 3*(As + Asr)/Ag  <=  0.7
    """
    if Ag <= 0.0:
        raise GeometryError(f"Ag must be positive, got {Ag}")
    return min(0.25 + 3.0 * (As + Asr) / Ag, 0.7)


def C3_coefficient(As: Inch2, Asr: Inch2, Ag: Inch2) -> Ratio:
    """Effective rigidity coefficient for a **filled** member.

    AISC 360-16, Eq. I2-13, Sect. I2.2b, p. 16.1-94::

        C3 = 0.45 + 3*(As + Asr)/Ag  <=  0.9

    Both the intercept and the cap are higher than Eq. I2-7's -- a filled tube
    holds its concrete together far better than encasement does, so more of the
    concrete stiffness may be counted.
    """
    if Ag <= 0.0:
        raise GeometryError(f"Ag must be positive, got {Ag}")
    return min(0.45 + 3.0 * (As + Asr) / Ag, 0.9)


def EIeff_encased(
    Es: Ksi, Is: Inch4, Isr: Inch4, C1: Ratio, Ec: Ksi, Ic: Inch4
) -> float:
    """Effective stiffness of an encased composite section.

    AISC 360-16, Eq. I2-6, Sect. I2.1b, p. 16.1-92::

        EIeff = Es*Is + Es*Isr + C1*Ec*Ic

    Note the reinforcement takes the **full** ``Es*Isr`` -- only the concrete is
    discounted by ``C1``.
    """
    if Es <= 0.0 or Ec <= 0.0:
        raise GeometryError(f"Es and Ec must be positive, got {Es}, {Ec}")
    return Es * Is + Es * Isr + C1 * Ec * Ic


def EIeff_filled(Es: Ksi, Is: Inch4, Isr: Inch4, C3: Ratio, Ec: Ksi, Ic: Inch4) -> float:
    """Effective stiffness of a filled composite section.

    AISC 360-16, Eq. I2-12, Sect. I2.2b, p. 16.1-94::

        EIeff = Es*Is + Es*Isr + C3*Ec*Ic

    Structurally identical to Eq. I2-6 with ``C3`` in place of ``C1``.
    """
    if Es <= 0.0 or Ec <= 0.0:
        raise GeometryError(f"Es and Ec must be positive, got {Es}, {Ec}")
    return Es * Is + Es * Isr + C3 * Ec * Ic


def Pe_composite(EIeff: float, Lc: Inch) -> Kip:
    """Elastic critical buckling load of a composite member.

    AISC 360-16, Eq. I2-5, Sect. I2.1b, p. 16.1-91::

        Pe = pi^2*(EIeff)/Lc^2
    """
    if EIeff <= 0.0:
        raise GeometryError(f"EIeff must be positive, got {EIeff}")
    if Lc <= 0.0:
        raise GeometryError(f"Lc must be positive, got {Lc}")
    return math.pi**2 * EIeff / Lc**2


def composite_compressive_strength(
    Pno: Kip, Pe: Kip, *, basis: Basis = Basis.LRFD, bare_steel_Pn: Kip = 0.0
) -> StrengthResult:
    """Nominal compressive strength of a composite member.

    AISC 360-16, Eqs. I2-2 and I2-3, Sect. I2.1b, p. 16.1-90::

        Pno/Pe <= 2.25:  Pn = Pno*(0.658^(Pno/Pe))            (I2-2)
        Pno/Pe >  2.25:  Pn = 0.877*Pe                        (I2-3)

    with ``phi_c = 0.75`` / ``Omega_c = 2.00`` -- **not** Chapter E's 0.90/1.67.

    The same 0.658, 0.877 and 2.25 as Eqs. E3-2/E3-3, but switching on
    ``Pno/Pe`` (two *loads*) rather than ``Fy/Fe`` (two *stresses*). They are the
    same curve: for a bare steel member ``Pno = Fy*Ag`` and ``Pe = Fe*Ag``, so
    the ratio is identical. For a composite member the areas differ between
    numerator and denominator, and only the load form is correct.

    Parameters
    ----------
    bare_steel_Pn:
        Sects. I2.1b and I2.2b both close with "the available compressive
        strength need not be less than that specified for the bare steel member,
        as required by Chapter E". A very lightly reinforced encased member with
        a slender concrete section can compute below its own steel core; pass
        the Chapter E value and it becomes the floor.
    """
    if Pno <= 0.0 or Pe <= 0.0:
        raise GeometryError(f"Pno and Pe must be positive, got {Pno}, {Pe}")

    ratio = Pno / Pe
    if ratio <= 2.25:
        Pn = Pno * math.pow(0.658, ratio)  # Eq. I2-2
        citation, note = cite("I2-2"), f"Pno/Pe = {ratio:.3f} <= 2.25, inelastic"
    else:
        Pn = 0.877 * Pe  # Eq. I2-3
        citation, note = cite("I2-3"), f"Pno/Pe = {ratio:.3f} > 2.25, elastic"

    if bare_steel_Pn > Pn:
        Pn = bare_steel_Pn
        note += "; raised to the bare steel Chapter E strength (Sect. I2.1b/I2.2b)"

    state = LimitStateResult(
        LimitState.FLEXURAL_BUCKLING, Pn, citation,
        detail={"Pno": Pno, "Pe": Pe, "Pno/Pe": ratio}, note=note,
        phi=PHI_C_COMPOSITE, omega=OMEGA_C_COMPOSITE,
    )
    return StrengthResult.build(
        "Pn", "kip", [state],
        phi=PHI_C_COMPOSITE, omega=OMEGA_C_COMPOSITE, basis=basis,
    )


def composite_tensile_strength(
    Fy: Ksi, As: Inch2, Fysr: Ksi = 0.0, Asr: Inch2 = 0.0, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Tensile strength of an encased or filled composite member.

    AISC 360-16, Eqs. I2-8 and I2-14, Sects. I2.1c and I2.2c, pp. 16.1-93 to
    16.1-95::

        Pn = Fy*As + Fysr*Asr

    with ``phi_t = 0.90`` / ``Omega_t = 1.67``.

    The **concrete contributes nothing** -- Sect. I1.2 states that "the tensile
    strength of the concrete shall be neglected in the determination of the
    nominal strength of composite members". The two equations are identical, and
    the Specification numbers them separately only because they sit in different
    sections.
    """
    if Fy <= 0.0 or As <= 0.0:
        raise GeometryError(f"Fy and As must be positive, got {Fy}, {As}")
    state = LimitStateResult(
        LimitState.TENSILE_YIELDING, Fy * As + Fysr * Asr, cite("I2-8"),
        detail={"Fy*As": Fy * As, "Fysr*Asr": Fysr * Asr},
        note="concrete contributes nothing in tension (Sect. I1.2)",
        phi=PHI_T_COMPOSITE, omega=OMEGA_T_COMPOSITE,
    )
    return StrengthResult.build(
        "Pn", "kip", [state],
        phi=PHI_T_COMPOSITE, omega=OMEGA_T_COMPOSITE, basis=basis,
    )


def Pp_filled(
    Fy: Ksi, As: Inch2, fc_prime: Ksi, Ac: Inch2, Asr: Inch2, Es: Ksi, Ec: Ksi,
    *, round_section: bool = False,
) -> Kip:
    """Squash load of a **compact** filled composite member.

    AISC 360-16, Eq. I2-9b, Sect. I2.2b, p. 16.1-93::

        Pp = Fy*As + C2*f'c*(Ac + Asr*Es/Ec)

    with ``C2 = 0.85`` for rectangular sections and **0.95 for round** ones -- a
    round tube confines its concrete more effectively, so more of ``f'c`` is
    available.

    The ``Asr*Es/Ec`` term transforms the reinforcement into equivalent concrete
    area rather than adding ``Fysr*Asr`` directly, which is what Eq. I2-4 does
    for encased members. The two treatments are genuinely different.
    """
    for name, value in (("Fy", Fy), ("As", As), ("fc_prime", fc_prime), ("Ac", Ac)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    if Ec <= 0.0:
        raise GeometryError(f"Ec must be positive, got {Ec}")
    C2 = 0.95 if round_section else 0.85
    return Fy * As + C2 * fc_prime * (Ac + Asr * Es / Ec)


def Py_filled(
    Fy: Ksi, As: Inch2, fc_prime: Ksi, Ac: Inch2, Asr: Inch2, Es: Ksi, Ec: Ksi
) -> Kip:
    """Yield load of a filled composite member, for the noncompact branch.

    AISC 360-16, Eq. I2-9d, Sect. I2.2b, p. 16.1-93::

        Py = Fy*As + 0.7*f'c*(Ac + Asr*Es/Ec)

    Note ``0.7`` here, not ``C2``. A noncompact tube buckles locally before the
    concrete reaches its rectangular-stress-block strength, so less of ``f'c``
    is counted regardless of section shape.
    """
    if Ec <= 0.0:
        raise GeometryError(f"Ec must be positive, got {Ec}")
    return Fy * As + 0.7 * fc_prime * (Ac + Asr * Es / Ec)


def Fcr_filled_rectangular(Es: Ksi, b_over_t: Ratio) -> Ksi:
    """Local buckling stress of a slender rectangular filled section.

    AISC 360-16, Eq. I2-10, Sect. I2.2b, p. 16.1-93::

        Fcr = 9*Es/(b/t)^2
    """
    if b_over_t <= 0.0:
        raise GeometryError(f"b/t must be positive, got {b_over_t}")
    return 9.0 * Es / b_over_t**2


def Fcr_filled_round(Fy: Ksi, D_over_t: Ratio, Es: Ksi) -> Ksi:
    """Local buckling stress of a slender round filled section.

    AISC 360-16, Eq. I2-11, Sect. I2.2b, p. 16.1-94::

        Fcr = 0.72*Fy / [(D/t)*(Fy/Es)]^0.2

    Unlike Eq. I2-10 this is not an inverse-square law -- the fifth-root
    exponent makes a round tube far less sensitive to slenderness, which is why
    Table I1.1a permits round filled sections to be much more slender.
    """
    if Fy <= 0.0 or D_over_t <= 0.0 or Es <= 0.0:
        raise GeometryError("Fy, D/t and Es must be positive")
    return 0.72 * Fy / math.pow(D_over_t * (Fy / Es), 0.2)


def Pno_filled(
    section_class: FlexuralSlenderness, Pp: Kip, Py: Kip = 0.0,
    *, lam: Ratio = 0.0, lam_p: Ratio = 0.0, lam_r: Ratio = 0.0,
    Fcr: Ksi = 0.0, As: Inch2 = 0.0, fc_prime: Ksi = 0.0, Ac: Inch2 = 0.0,
    Asr: Inch2 = 0.0, Es: Ksi = 29000.0, Ec: Ksi = 0.0,
) -> tuple[Kip, str]:
    """Squash load of a filled composite member, by local-buckling class.

    AISC 360-16, Eqs. I2-9a, I2-9c and I2-9e, Sect. I2.2b, p. 16.1-93::

        compact:     Pno = Pp                                     (I2-9a)
        noncompact:  Pno = Pp - (Pp - Py)*((lam - lam_p)/(lam_r - lam_p))^2
                                                                  (I2-9c)
        slender:     Pno = Fcr*As + 0.7*f'c*(Ac + Asr*Es/Ec)       (I2-9e)

    Eq. I2-9c interpolates on the **square** of the slenderness fraction, not
    linearly. Every other noncompact transition in the Specification -- F3-1,
    F4-13, F7-2, I3-3b -- is linear; this one is not, and using a linear
    interpolation here is unconservative across the whole band.

    Returns ``(Pno, note)``.
    """
    if section_class is FlexuralSlenderness.COMPACT:
        return Pp, "Eq. I2-9a: compact, Pno = Pp"

    if section_class is FlexuralSlenderness.NONCOMPACT:
        if lam_r <= lam_p:
            raise AISC360Error(f"lambda_r ({lam_r}) must exceed lambda_p ({lam_p})")
        fraction = (lam - lam_p) / (lam_r - lam_p)
        fraction = min(max(fraction, 0.0), 1.0)
        # Eq. I2-9c -- the fraction is SQUARED, unlike every other noncompact
        # transition in the Specification.
        return (
            Pp - (Pp - Py) * fraction**2,
            f"Eq. I2-9c: noncompact, SQUARED interpolation at fraction {fraction:.4f}",
        )

    if Ec <= 0.0:
        raise GeometryError(f"Ec must be positive for the slender branch, got {Ec}")
    return (
        Fcr * As + 0.7 * fc_prime * (Ac + Asr * Es / Ec),  # Eq. I2-9e
        "Eq. I2-9e: slender, Fcr replaces Fy and the concrete drops to 0.7*f'c",
    )


# ===========================================================================
# Sect. I3 -- Flexure
# ===========================================================================
def horizontal_shear_positive(
    fc_prime: Ksi, Ac: Inch2, Fy: Ksi, As: Inch2, sum_Qn: Kip
) -> tuple[Kip, str]:
    """Total horizontal shear in the positive moment region of a composite beam.

    AISC 360-16, Eqs. I3-1a, I3-1b and I3-1c, Sect. I3.1d, p. 16.1-97. The
    **lowest** of::

        concrete crushing:           V' = 0.85*f'c*Ac              (I3-1a)
        tensile yielding of steel:   V' = Fy*As                    (I3-1b)
        shear strength of anchors:   V' = sum(Qn)                  (I3-1c)

    Which one governs decides the design: if the anchors govern the beam is
    *partially composite*, and its flexural strength must be computed from the
    anchor force rather than from full composite action.
    """
    candidates = {
        "I3-1a": (0.85 * fc_prime * Ac, "concrete crushing"),
        "I3-1b": (Fy * As, "tensile yielding of the steel section"),
        "I3-1c": (sum_Qn, "shear strength of the anchors -- PARTIALLY COMPOSITE"),
    }
    equation = min(candidates, key=lambda k: candidates[k][0])
    value, description = candidates[equation]
    return value, f"Eq. {equation}: {description}"


def horizontal_shear_negative(Fysr: Ksi, Asr: Inch2, sum_Qn: Kip) -> tuple[Kip, str]:
    """Total horizontal shear in the negative moment region.

    AISC 360-16, Eqs. I3-2a and I3-2b, Sect. I3.1e, p. 16.1-97. The lower of::

        tensile yielding of slab reinforcement:  V' = Fysr*Asr     (I3-2a)
        shear strength of the anchors:           V' = sum(Qn)      (I3-2b)

    The concrete does not appear at all -- in negative bending the slab is in
    tension and Sect. I1.2 neglects concrete tension entirely.
    """
    if Fysr * Asr <= sum_Qn:
        return Fysr * Asr, "Eq. I3-2a: tensile yielding of the slab reinforcement"
    return sum_Qn, "Eq. I3-2b: shear strength of the anchors"


def filled_flexural_strength(
    section_class: FlexuralSlenderness, Mp: KipIn, My: KipIn = 0.0,
    *, lam: Ratio = 0.0, lam_p: Ratio = 0.0, lam_r: Ratio = 0.0,
    Mcr: KipIn = 0.0, basis: Basis = Basis.LRFD,
) -> StrengthResult:
    """Flexural strength of a filled composite member.

    AISC 360-16, Eqs. I3-3a and I3-3b, Sect. I3.4b, p. 16.1-98::

        compact:     Mn = Mp                                       (I3-3a)
        noncompact:  Mn = Mp - (Mp - My)*(lam - lam_p)/(lam_r - lam_p)
                                                                   (I3-3b)
        slender:     Mn = Mcr, the first-yield moment with the concrete
                     compressive stress limited to 0.70*f'c

    with ``phi_b = 0.90`` / ``Omega_b = 1.67``.

    Eq. I3-3b interpolates **linearly** -- unlike Eq. I2-9c for the same
    section in compression, which squares the fraction. The same member has a
    linear flexural transition and a quadratic compressive one.
    """
    if Mp <= 0.0:
        raise GeometryError(f"Mp must be positive, got {Mp}")

    if section_class is FlexuralSlenderness.COMPACT:
        Mn, citation, note = Mp, cite("I3-3a"), "compact: Mn = Mp"
    elif section_class is FlexuralSlenderness.NONCOMPACT:
        if lam_r <= lam_p:
            raise AISC360Error(f"lambda_r ({lam_r}) must exceed lambda_p ({lam_p})")
        fraction = min(max((lam - lam_p) / (lam_r - lam_p), 0.0), 1.0)
        Mn = Mp - (Mp - My) * fraction  # Eq. I3-3b -- LINEAR, unlike Eq. I2-9c
        citation = cite("I3-3b")
        note = f"noncompact: LINEAR interpolation at fraction {fraction:.4f}"
    else:
        if Mcr <= 0.0:
            raise AISC360Error(
                "a slender filled section needs Mcr -- the first-yield moment with "
                "the concrete compressive stress limited to 0.70*f'c (Sect. I3.4b)"
            )
        Mn, citation, note = Mcr, cite("I3-3b"), "slender: Mn = Mcr at 0.70*f'c"

    state = LimitStateResult(
        LimitState.YIELDING, Mn, citation, detail={"Mp": Mp, "My": My}, note=note,
        phi=PHI_B_COMPOSITE, omega=OMEGA_B_COMPOSITE,
    )
    return StrengthResult.build(
        "Mn", "kip-in.", [state],
        phi=PHI_B_COMPOSITE, omega=OMEGA_B_COMPOSITE, basis=basis,
    )


# ===========================================================================
# Sect. I5 -- Combined flexure and axial force
# ===========================================================================
def csr_ratio(As: Inch2, Fy: Ksi, Asr: Inch2, Fyr: Ksi, Ac: Inch2, fc_prime: Ksi) -> Ratio:
    """Steel-to-concrete strength ratio for a filled composite member.

    AISC 360-16, Eq. I5-2, Sect. I5, p. 16.1-101::

        csr = (As*Fy + Asr*Fyr)/(Ac*f'c)

    Selects the Table I5.1 row and the ``cm`` branch. A steel-dominated section
    (high ``csr``) behaves more like a bare steel beam-column and gets a
    correspondingly milder interaction.
    """
    if Ac <= 0.0 or fc_prime <= 0.0:
        raise GeometryError(f"Ac and f'c must be positive, got {Ac}, {fc_prime}")
    return (As * Fy + Asr * Fyr) / (Ac * fc_prime)


def cp_cm_coefficients(csr: Ratio, *, round_section: bool = False) -> tuple[float, float]:
    """Interaction coefficients ``(cp, cm)`` from Table I5.1, p. 16.1-100.

    Rectangular filled members::

        cp = 0.17/csr^0.4
        cm = 1.06/csr^0.11 >= 1.0    when csr >= 0.5
        cm = 0.90/csr^0.36 <= 1.67   when csr <  0.5

    Round HSS filled members::

        cp = 0.27/csr^0.4
        cm = 1.10/csr^0.08 >= 1.0    when csr >= 0.5
        cm = 0.95/csr^0.32 <= 1.67   when csr <  0.5

    Note the bounds run in **opposite directions** between the two ``cm``
    branches: a floor of 1.0 above ``csr = 0.5`` and a ceiling of 1.67 below it.
    They are not the same expression evaluated in two ranges.
    """
    if csr <= 0.0:
        raise GeometryError(f"csr must be positive, got {csr}")

    if round_section:
        cp = 0.27 / math.pow(csr, 0.4)
        cm = (
            max(1.10 / math.pow(csr, 0.08), 1.0) if csr >= 0.5
            else min(0.95 / math.pow(csr, 0.32), 1.67)
        )
    else:
        cp = 0.17 / math.pow(csr, 0.4)
        cm = (
            max(1.06 / math.pow(csr, 0.11), 1.0) if csr >= 0.5
            else min(0.90 / math.pow(csr, 0.36), 1.67)
        )
    return cp, cm


def composite_interaction(
    Pr: Kip, Pc: Kip, Mr: KipIn, Mc: KipIn, cp: float, cm: float
) -> InteractionResult:
    """Combined axial force and flexure in a filled composite beam-column.

    AISC 360-16, Eqs. I5-1a and I5-1b, Sect. I5, p. 16.1-99::

        Pr/Pc >= cp:  Pr/Pc + ((1 - cp)/cm)*(Mr/Mc)  <= 1.0        (I5-1a)
        Pr/Pc <  cp:  ((1 - cm)/cp)*(Pr/Pc) + Mr/Mc  <= 1.0        (I5-1b)

    Sect. I5 also permits Eqs. H1-1a/H1-1b or the Sect. I1.2d method; this is
    the sharpest of the three for a filled member.

    The branch point is ``cp``, not the fixed 0.2 of Eq. H1-1a -- and ``cp``
    depends on the section through ``csr``. Two filled columns of different
    steel ratio switch branches at different axial loads.

    Note that ``cm`` can exceed 1.0, which makes ``(1 - cm)/cp`` **negative** in
    Eq. I5-1b -- so a small axial load genuinely *reduces* the interaction
    ratio. That is deliberate: modest compression closes concrete cracks and
    raises the flexural capacity of a filled section.
    """
    if Pc <= 0.0 or Mc <= 0.0:
        raise GeometryError(f"Pc and Mc must be positive, got {Pc}, {Mc}")
    if cp <= 0.0 or cm <= 0.0:
        raise GeometryError(f"cp and cm must be positive, got {cp}, {cm}")

    axial = abs(Pr) / Pc
    flexure = abs(Mr) / Mc

    if axial >= cp:
        ratio = axial + ((1.0 - cp) / cm) * flexure  # Eq. I5-1a
        citation, note = cite("I5-1a"), f"Pr/Pc = {axial:.4f} >= cp = {cp:.4f}"
    else:
        ratio = ((1.0 - cm) / cp) * axial + flexure  # Eq. I5-1b
        citation = cite("I5-1b")
        note = f"Pr/Pc = {axial:.4f} < cp = {cp:.4f}"
        if cm > 1.0:
            note += "; cm > 1 makes the axial term NEGATIVE -- compression helps here"

    return InteractionResult(
        ratio=max(ratio, 0.0), citation=citation,
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE,
        terms={"Pr/Pc": axial, "Mr/Mc": flexure, "cp": cp, "cm": cm},
        note=note,
    )


# ===========================================================================
# Sect. I6 -- Load transfer
# ===========================================================================
def force_to_concrete(Pr: Kip, Fy: Ksi, As: Inch2, Pno: Kip) -> Kip:
    """Longitudinal shear to transfer when the load lands on the steel section.

    AISC 360-16, Eq. I6-1, Sect. I6.2a, p. 16.1-102::

        V'r = Pr*(1 - Fy*As/Pno)

    The bracket is the *concrete's* share -- the steel already has its own
    portion, so only the remainder needs transferring across the interface.
    """
    if Pno <= 0.0:
        raise GeometryError(f"Pno must be positive, got {Pno}")
    return abs(Pr) * (1.0 - Fy * As / Pno)


def force_to_steel(Pr: Kip, F: Ksi, As: Inch2, Pno: Kip, *, slender: bool = False) -> Kip:
    """Longitudinal shear to transfer when the load lands on the concrete.

    AISC 360-16, Eqs. I6-2a and I6-2b, Sect. I6.2b, p. 16.1-102::

        compact or noncompact:  V'r = Pr*(Fy*As/Pno)               (I6-2a)
        slender filled:         V'r = Pr*(Fcr*As/Pno)              (I6-2b)

    The complement of Eq. I6-1: here the *steel's* share is what must cross the
    interface. For a slender filled section the steel cannot reach ``Fy`` before
    it buckles locally, so ``Fcr`` from Eq. I2-10 or I2-11 replaces it.
    """
    if Pno <= 0.0:
        raise GeometryError(f"Pno must be positive, got {Pno}")
    del slender  # the caller chooses which stress to pass; the form is identical
    return abs(Pr) * (F * As / Pno)


def direct_bearing_strength(
    fc_prime: Ksi, A1: Inch2, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Direct bearing strength of concrete in a composite member.

    AISC 360-16, Eq. I6-3, Sect. I6.3a, p. 16.1-103::

        Rn = 1.7*f'c*A1

    with ``phi_B = 0.65`` / ``Omega_B = 2.31``.

    Note the ``1.7`` -- twice Sect. J8's ``0.85`` for bearing on a concrete
    support. Concrete inside a composite member is confined by the surrounding
    section, so it carries the full ``sqrt(A2/A1) = 2`` confinement bonus that
    Eq. J8-2 caps at. The User Note to Sect. I6.2 says as much: for filled
    members "the term ``sqrt(A2/A1)`` in Equation J8-2 may be taken equal to
    2.0 due to confinement effects".
    """
    if fc_prime <= 0.0 or A1 <= 0.0:
        raise GeometryError(f"f'c and A1 must be positive, got {fc_prime}, {A1}")
    state = LimitStateResult(
        LimitState.CONCRETE_BEARING, 1.7 * fc_prime * A1, cite("I6-3"),
        detail={"f'c": fc_prime, "A1": A1},
        note="1.7, not Sect. J8's 0.85 -- the concrete is confined by the section",
        phi=PHI_BEARING, omega=OMEGA_BEARING,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_BEARING, omega=OMEGA_BEARING, basis=basis
    )


def shear_connection_strength(sum_Qcv: Kip) -> tuple[Kip, str]:
    """Load-transfer strength by shear connection.

    AISC 360-16, Eq. I6-4, Sect. I6.3b, p. 16.1-103::

        Rc = sum(Qcv)

    ``Qcv`` is the **available** strength of each anchor -- ``phi*Qnv`` or
    ``Qnv/Omega`` from Sect. I8.3a -- not the nominal. Eq. I6-4 is therefore the
    one place in this library that sums already-factored quantities, which is
    why it returns a plain value rather than a
    :class:`~pyaisc360.core.result.StrengthResult`: there is no further factor
    to apply.

    Only anchors "placed within the load introduction length as defined in
    Section I6.4" count. Anchors outside it are not transferring the
    introduction force and must not be included.
    """
    if sum_Qcv < 0.0:
        raise GeometryError(f"the summed anchor strength must be non-negative, got {sum_Qcv}")
    return sum_Qcv, (
        "Eq. I6-4: sum of AVAILABLE anchor strengths within the load introduction "
        "length -- already factored, no further phi applies"
    )


def bond_stress(t: Inch, dimension: Inch, *, round_section: bool = False) -> Ksi:
    """Nominal bond stress between steel and concrete in a filled member.

    AISC 360-16, Sect. I6.3c, p. 16.1-103::

        rectangular:  Fin = 12*t/H^2  <= 0.1 ksi
        round:        Fin = 30*t/D^2  <= 0.2 ksi

    Both are **dimensional** expressions in inches and ksi -- the metric forms
    use 2100 and 5300 with caps of 0.7 and 1.4 MPa, which are not unit
    conversions of these.

    The caps bite for almost any real section: a 6-inch square tube with a
    quarter-inch wall gives ``12*0.25/36 = 0.083`` ksi, just under; anything
    smaller or thicker is capped.
    """
    if t <= 0.0 or dimension <= 0.0:
        raise GeometryError("t and the section dimension must be positive")
    if round_section:
        return min(30.0 * t / dimension**2, 0.2)
    return min(12.0 * t / dimension**2, 0.1)


def bond_strength(
    perimeter: Inch, Lin: Inch, Fin: Ksi, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Bond strength of a filled composite member.

    AISC 360-16, Eq. I6-5, Sect. I6.3c, p. 16.1-103::

        Rn = pb*Lin*Fin

    with ``phi = 0.50`` / ``Omega = 3.00`` -- **the lowest resistance factor in
    the Specification**. Bond is the least reliable of the three load-transfer
    mechanisms, and Sect. I6.3 forbids it entirely for encased members.

    Sect. I6.3 also states the mechanisms "shall not be superimposed" -- use the
    largest of direct bearing, shear connection and bond, never their sum.
    """
    if perimeter <= 0.0 or Lin <= 0.0 or Fin <= 0.0:
        raise GeometryError("perimeter, Lin and Fin must be positive")
    state = LimitStateResult(
        LimitState.YIELDING, perimeter * Lin * Fin, cite("I6-5"),
        detail={"pb": perimeter, "Lin": Lin, "Fin": Fin},
        note=(
            "phi = 0.50, the lowest in the Specification; filled members only, and "
            "Sect. I6.3 forbids superimposing the transfer mechanisms"
        ),
        phi=PHI_BOND, omega=OMEGA_BOND,
    )
    return StrengthResult.build(
        "Rn", "kip", [state], phi=PHI_BOND, omega=OMEGA_BOND, basis=basis
    )


# ===========================================================================
# Sect. I8 -- Steel anchors
# ===========================================================================
def Rg_Rp(
    orientation: DeckOrientation,
    *,
    anchors_per_rib: int = 1,
    rib_width_to_depth: float = 2.0,
    emid_ht_at_least_2in: bool = True,
) -> tuple[float, float, str]:
    """Group and position effect factors ``(Rg, Rp)`` for Eq. I8-1.

    AISC 360-16, Sect. I8.2a, pp. 16.1-105 to 16.1-106::

        no decking:                              Rg = 1.0,  Rp = 0.75
        deck parallel,   wr/hr >= 1.5:           Rg = 1.0,  Rp = 0.75
        deck parallel,   wr/hr <  1.5:           Rg = 0.85, Rp = 0.75
        deck perpendicular, 1 anchor per rib:    Rg = 1.0,  Rp = 0.6
        deck perpendicular, 2 anchors per rib:   Rg = 0.85, Rp = 0.6
        deck perpendicular, 3+ anchors per rib:  Rg = 0.7,  Rp = 0.6

    ``Rp = 0.6`` may be raised to 0.75 when ``emid-ht >= 2 in.`` -- the distance
    from the stud shank to the deck web at mid-rib-height, in the load-bearing
    direction. That single detail is worth 25% of the anchor's strength and is
    the most commonly missed input in Sect. I8.

    Crowding studs into a perpendicular rib is doubly penalised: three anchors
    per rib give ``Rg*Rp = 0.7*0.6 = 0.42`` against ``1.0*0.75 = 0.75`` with no
    deck -- a 44% loss per anchor.
    """
    if anchors_per_rib < 1:
        raise GeometryError(f"there must be at least one anchor, got {anchors_per_rib}")

    if orientation is DeckOrientation.NONE:
        return 1.0, 0.75, "no decking"

    if orientation is DeckOrientation.PARALLEL:
        Rg = 1.0 if rib_width_to_depth >= 1.5 else 0.85
        return Rg, 0.75, f"deck parallel, wr/hr = {rib_width_to_depth:g}"

    Rg = {1: 1.0, 2: 0.85}.get(anchors_per_rib, 0.7)
    Rp = 0.75 if emid_ht_at_least_2in else 0.6
    note = (
        f"deck perpendicular, {anchors_per_rib} anchor(s) per rib; "
        f"emid-ht {'>=' if emid_ht_at_least_2in else '<'} 2 in."
    )
    return Rg, Rp, note


def stud_anchor_shear_in_beam(
    Asa: Inch2, fc_prime: Ksi, Ec: Ksi, Rg: float, Rp: float, Fu: Ksi
) -> tuple[Kip, str]:
    """Nominal shear strength of one steel headed stud anchor in a composite beam.

    AISC 360-16, Eq. I8-1, Sect. I8.2a, p. 16.1-105::

        Qn = 0.5*Asa*sqrt(f'c*Ec)  <=  Rg*Rp*Asa*Fu

    The first term is **concrete crushing around the stud**; the cap is the
    **stud shearing off**. Which governs is worth knowing: in normal-weight
    concrete of 4 ksi or above the cap almost always governs, so the anchor is
    limited by ``Rg*Rp`` -- which means the deck detail, not the concrete grade,
    sets the strength.

    Returns ``(Qn, governing)``.
    """
    for name, value in (("Asa", Asa), ("f'c", fc_prime), ("Ec", Ec), ("Fu", Fu)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    concrete = 0.5 * Asa * math.sqrt(fc_prime * Ec)
    steel = Rg * Rp * Asa * Fu
    if concrete <= steel:
        return concrete, "Eq. I8-1: concrete crushing around the stud governs"
    return steel, f"Eq. I8-1 cap: stud shear governs at Rg*Rp = {Rg * Rp:.3f}"


def channel_anchor_shear(
    tf: Inch, tw: Inch, la: Inch, fc_prime: Ksi, Ec: Ksi
) -> Kip:
    """Nominal shear strength of one hot-rolled channel anchor.

    AISC 360-16, Eq. I8-2, Sect. I8.2b, p. 16.1-106::

        Qn = 0.3*(tf + 0.5*tw)*la*sqrt(f'c*Ec)

    Note ``tf + 0.5*tw`` -- the flange thickness plus **half** the web, not the
    sum. And unlike Eq. I8-1 there is no ``Rg*Rp`` and no steel-strength cap:
    Sect. I8.2b instead requires the weld to the beam flange to develop ``Qn``
    directly, "considering eccentricity on the anchor".
    """
    for name, value in (("tf", tf), ("tw", tw), ("la", la), ("f'c", fc_prime), ("Ec", Ec)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    return 0.3 * (tf + 0.5 * tw) * la * math.sqrt(fc_prime * Ec)


def stud_shear_strength(
    Fu: Ksi, Asa: Inch2, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Shear strength of one stud anchor in a composite component.

    AISC 360-16, Eq. I8-3, Sect. I8.3a, p. 16.1-109::

        Qnv = Fu*Asa

    with ``phi_v = 0.65`` / ``Omega_v = 2.31``.

    Applies "where concrete breakout strength in shear is **not** an applicable
    limit state" -- which Sect. I8.3a permits only when anchor reinforcement is
    developed per ACI 318 on both sides of the breakout surface, or the
    component is otherwise detailed to preclude breakout. Breakout itself is an
    ACI 318 calculation and is outside this Specification.
    """
    if Fu <= 0.0 or Asa <= 0.0:
        raise GeometryError(f"Fu and Asa must be positive, got {Fu}, {Asa}")
    state = LimitStateResult(
        LimitState.BOLT_SHEAR, Fu * Asa, cite("I8-3"),
        detail={"Fu": Fu, "Asa": Asa},
        note=(
            "valid only where concrete breakout in shear is NOT an applicable limit "
            "state -- breakout is an ACI 318 calculation, outside this Specification"
        ),
        phi=PHI_V_ANCHOR, omega=OMEGA_V_ANCHOR,
    )
    return StrengthResult.build(
        "Qnv", "kip", [state], phi=PHI_V_ANCHOR, omega=OMEGA_V_ANCHOR, basis=basis
    )


def stud_tensile_strength(
    Fu: Ksi, Asa: Inch2, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Tensile strength of one stud anchor in a composite component.

    AISC 360-16, Eq. I8-4, Sect. I8.3b, p. 16.1-109::

        Qnt = Fu*Asa

    with ``phi_t = 0.75`` / ``Omega_t = 2.00``.

    The same product as Eq. I8-3 but a **different factor pair** -- 0.75/2.00
    against 0.65/2.31. So a stud is nominally as strong in tension as in shear
    but has more available strength in tension, which is the opposite of the
    intuition from bolts.

    Conditional on edge distance >= 1.5 times the stud height and spacing >= 3
    times the stud height, both measured to the top of the head.
    """
    if Fu <= 0.0 or Asa <= 0.0:
        raise GeometryError(f"Fu and Asa must be positive, got {Fu}, {Asa}")
    state = LimitStateResult(
        LimitState.BOLT_TENSION, Fu * Asa, cite("I8-4"),
        detail={"Fu": Fu, "Asa": Asa},
        note="requires edge distance >= 1.5*h and spacing >= 3*h to the top of the head",
        phi=PHI_T_ANCHOR, omega=OMEGA_T_ANCHOR,
    )
    return StrengthResult.build(
        "Qnt", "kip", [state], phi=PHI_T_ANCHOR, omega=OMEGA_T_ANCHOR, basis=basis
    )


def anchor_interaction(Qrt: Kip, Qct: Kip, Qrv: Kip, Qcv: Kip) -> InteractionResult:
    """Combined shear and tension on one steel headed stud anchor.

    AISC 360-16, Eq. I8-5, Sect. I8.3c, p. 16.1-110::

        (Qrt/Qct)^(5/3) + (Qrv/Qcv)^(5/3)  <=  1.0

    The **5/3 exponent** is unique in the Specification -- everything else uses
    linear terms, squares, or the 8/9 of Eq. H1-1a. It makes the interaction
    curve bulge outward: two effects at 60% of capacity each give
    ``2*0.6^1.667 = 0.85``, comfortably adequate, where a linear rule would give
    1.20 and a circular one 0.72.

    Treating it as linear is needlessly conservative; treating it as circular is
    unconservative.
    """
    if Qct <= 0.0 or Qcv <= 0.0:
        raise GeometryError(f"Qct and Qcv must be positive, got {Qct}, {Qcv}")

    tension = abs(Qrt) / Qct
    shear = abs(Qrv) / Qcv
    exponent = 5.0 / 3.0
    return InteractionResult(
        ratio=math.pow(tension, exponent) + math.pow(shear, exponent),
        citation=cite("I8-5"),
        limit_state=LimitState.BOLT_TENSION,
        terms={
            "Qrt/Qct": tension,
            "Qrv/Qcv": shear,
            "(Qrt/Qct)^(5/3)": math.pow(tension, exponent),
            "(Qrv/Qcv)^(5/3)": math.pow(shear, exponent),
        },
        note="the 5/3 exponent is unique in the Specification",
    )
