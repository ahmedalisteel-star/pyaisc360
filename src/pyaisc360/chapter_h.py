"""Chapter H -- Design of Members for Combined Forces and Torsion.

Covers Sects. H1 through H4, pp. 16.1-77 to 16.1-85.

Chapter H consumes the rest of the member chapters rather than replacing them:
``Pc`` comes from Chapter E (compression) or Sect. D2 (tension), ``Mc`` from
Chapter F, ``Vc`` from Chapter G, and ``Tc`` from Sect. H3.1. Every provision
here is an **inequality on those available strengths**, so the functions return
:class:`~pyaisc360.core.result.InteractionResult` -- a ratio, not a strength.

The one exception is Sect. H3.1, which *is* a strength: the torsional capacity
of an HSS, returned as a :class:`~pyaisc360.core.result.StrengthResult`.

Because the factors are already inside ``Pc``, ``Mc`` and ``Vc``, **the caller
must supply demands and capacities on the same basis**. Mixing an LRFD demand
with an ASD capacity produces a number that means nothing, and no amount of
checking inside this module can detect it -- so ``basis`` is carried on the
member and echoed onto the result for the calculation sheet.
"""

from __future__ import annotations

import math

from .core.citations import cite
from .core.config import Basis
from .core.enums import LimitState
from .core.exceptions import GeometryError, OutOfScopeError
from .core.result import InteractionResult, LimitStateResult, StrengthResult
from .core.units import Dimensionless, Inch, Inch3, Inch4, Kip, KipIn, Ksi, Ratio

__all__ = [
    "PHI_T_TORSION",
    "OMEGA_T_TORSION",
    "ALPHA_LRFD",
    "ALPHA_ASD",
    "H1_BRANCH_THRESHOLD",
    "TORSION_NEGLECT_THRESHOLD",
    "alpha_for",
    "combined_axial_flexure",
    "Pey",
    "cb_tension_multiplier",
    "in_plane_instability",
    "out_of_plane_buckling",
    "h1_3_applicable",
    "combined_stress_ratio",
    "round_hss_torsional_constant",
    "rectangular_hss_torsional_constant",
    "round_hss_torsional_stress",
    "rectangular_hss_torsional_stress",
    "hss_torsional_strength",
    "torsion_may_be_neglected",
    "hss_combined_torsion",
    "non_hss_torsional_stress",
    "flange_rupture_interaction",
]

#: Sect. H3.1, p. 16.1-81 -- HSS torsional yielding and buckling.
PHI_T_TORSION: float = 0.90
OMEGA_T_TORSION: float = 1.67

#: The ASD force-level adjustment. Second-order and stability provisions are
#: calibrated on ultimate-level forces, so an ASD demand -- which is at service
#: level -- must be scaled by 1.6 before entering them. Appears in Eqs. H1-2,
#: A-7-1, A-7-2, A-8-3 and A-8-6, always with the same value.
ALPHA_LRFD: float = 1.0
ALPHA_ASD: float = 1.6

#: Sect. H1.1, p. 16.1-77: the axial ratio selecting Eq. H1-1a or H1-1b.
H1_BRANCH_THRESHOLD: Ratio = 0.2

#: Sect. H3.2, p. 16.1-83: below this Tr/Tc, torsion may be neglected entirely.
TORSION_NEGLECT_THRESHOLD: Ratio = 0.20


def alpha_for(basis: Basis) -> float:
    """``alpha`` = 1.0 (LRFD) or 1.6 (ASD).

    AISC 360-16, Sects. H1.2, A-7.3, A-8.2.1 and A-8.2.2.
    """
    return ALPHA_LRFD if basis.is_lrfd else ALPHA_ASD


# ===========================================================================
# Sect. H1.1 / H1.2 -- the general interaction equations
# ===========================================================================
def combined_axial_flexure(
    Pr: Kip,
    Pc: Kip,
    Mrx: KipIn = 0.0,
    Mcx: KipIn = 0.0,
    Mry: KipIn = 0.0,
    Mcy: KipIn = 0.0,
) -> InteractionResult:
    """Combined axial force and flexure, Eqs. H1-1a and H1-1b.

    AISC 360-16, Sect. H1.1, p. 16.1-77::

        Pr/Pc >= 0.2:  Pr/Pc     + (8/9)*(Mrx/Mcx + Mry/Mcy) <= 1.0   (H1-1a)
        Pr/Pc <  0.2:  Pr/(2*Pc) +       (Mrx/Mcx + Mry/Mcy) <= 1.0   (H1-1b)

    Serves both Sect. H1.1 (flexure and **compression**, with ``Pc`` from
    Chapter E) and Sect. H1.2 (flexure and **tension**, with ``Pc`` from
    Sect. D2). The equations are identical; only the source of ``Pc`` changes.

    Notes
    -----
    The two branches describe one continuous *interaction curve* but not a
    continuous *utilization ratio*. Setting each equation to 1.0 and solving at
    ``Pr/Pc = 0.2`` gives ``Mr/Mc = 0.9`` from both -- the curve has a kink
    there but no gap. The computed ratio, however, steps by ``0.1 - (Mr/Mc)/9``
    as ``Pr/Pc`` crosses 0.2, which is zero only at ``Mr/Mc = 0.9``.

    That is not an error and not something to smooth: both expressions are
    valid adequacy measures, they agree exactly on the boundary curve, and the
    ratio is simply not a normalised distance to it. A member at
    ``Pr/Pc = 0.199`` and one at ``0.201`` can report different ratios while
    both being correctly assessed.

    Parameters
    ----------
    Pr, Pc:
        Required and available axial strength, same basis.
    Mrx, Mcx, Mry, Mcy:
        Required and available flexural strengths about each axis. A zero
        required moment contributes nothing and its capacity may be omitted.
    """
    if Pc <= 0.0:
        raise GeometryError(f"available axial strength must be positive, got {Pc}")
    if Pr < 0.0:
        raise GeometryError(f"required axial strength must be non-negative, got {Pr}")

    flexure = 0.0
    terms: dict[str, float] = {}
    for label, Mr, Mc in (("x", Mrx, Mcx), ("y", Mry, Mcy)):
        if Mr == 0.0:
            continue
        if Mc <= 0.0:
            raise GeometryError(
                f"Mr{label} = {Mr} was supplied with Mc{label} = {Mc}; an available "
                "flexural strength is needed wherever a required moment acts"
            )
        contribution = abs(Mr) / Mc
        terms[f"Mr{label}/Mc{label}"] = contribution
        flexure += contribution

    axial = Pr / Pc
    terms["Pr/Pc"] = axial

    if axial >= H1_BRANCH_THRESHOLD:
        ratio = axial + (8.0 / 9.0) * flexure  # Eq. H1-1a
        citation, note = cite("H1-1a"), "Pr/Pc >= 0.2"
    else:
        ratio = axial / 2.0 + flexure  # Eq. H1-1b
        citation, note = cite("H1-1b"), "Pr/Pc < 0.2"

    return InteractionResult(
        ratio=ratio, citation=citation,
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE, terms=terms, note=note,
    )


def Pey(E: Ksi, Iy: Inch4, Lb: Inch) -> Kip:
    """Elastic minor-axis buckling load over the unbraced length.

    AISC 360-16, Eq. H1-2, Sect. H1.2, p. 16.1-79::

        Pey = pi^2*E*Iy / Lb^2
    """
    if Iy <= 0.0 or Lb <= 0.0:
        raise GeometryError(f"Iy and Lb must be positive, got {Iy}, {Lb}")
    return math.pi**2 * E * Iy / Lb**2


def cb_tension_multiplier(Pr: Kip, Pey_value: Kip, basis: Basis) -> Dimensionless:
    """Factor by which ``Cb`` may be increased for concurrent axial tension.

    AISC 360-16, Sect. H1.2, p. 16.1-79::

        multiplier = sqrt(1 + alpha*Pr/Pey),   alpha = 1.0 (LRFD), 1.6 (ASD)

    Permitted for **doubly symmetric members only**. Axial tension straightens
    the member and stiffens it against lateral-torsional buckling, so ``Cb``
    from Chapter F may be raised. It is an allowance, never a requirement --
    ignoring it is always conservative.

    ``Pr`` here is the axial **tension**; passing a compression force would
    increase ``Cb`` when it should be reduced, so a negative value is rejected.
    """
    if Pr < 0.0:
        raise GeometryError(
            f"Pr = {Pr}: Sect. H1.2's Cb increase applies to axial TENSION acting "
            "concurrently with flexure. Compression does not stiffen the member "
            "against LTB and must not use this multiplier."
        )
    if Pey_value <= 0.0:
        raise GeometryError(f"Pey must be positive, got {Pey_value}")
    return math.sqrt(1.0 + alpha_for(basis) * Pr / Pey_value)


# ===========================================================================
# Sect. H1.3 -- the separated in-plane / out-of-plane check
# ===========================================================================
def h1_3_applicable(
    Lcz: Inch, Lcy: Inch, Mry: KipIn, Mcy: KipIn, *, doubly_symmetric_rolled_compact: bool
) -> tuple[bool, str]:
    """Whether the separated treatment of Sect. H1.3 may be used.

    AISC 360-16, Sect. H1.3, p. 16.1-79. All of the following are required:

    * doubly symmetric, rolled, compact member;
    * ``Lcz <= Lcy`` -- torsional effective length not exceeding the y-axis one;
    * moments primarily about the major axis, specifically ``Mry/Mcy < 0.05``.

    Sect. H1.3 states the last as "For members with ``Mry/Mcy >= 0.05``, the
    provisions of Section H1.1 shall be followed" -- so it is a hard cut-off,
    not a guideline. Returns ``(applicable, reason)``.
    """
    if not doubly_symmetric_rolled_compact:
        return False, "Sect. H1.3 applies to doubly symmetric rolled compact members only"
    if Lcz > Lcy:
        return False, f"Lcz = {Lcz:g} exceeds Lcy = {Lcy:g}; Sect. H1.3 requires Lcz <= Lcy"
    if Mcy > 0.0 and abs(Mry) / Mcy >= 0.05:
        return False, (
            f"Mry/Mcy = {abs(Mry) / Mcy:.4f} reaches 0.05; Sect. H1.3 directs that "
            "Sect. H1.1 be followed"
        )
    return True, "Sect. H1.3 separated in-plane / out-of-plane check permitted"


def in_plane_instability(Pr: Kip, Pcx: Kip, Mrx: KipIn, Mcx_yielding: KipIn) -> InteractionResult:
    """Sect. H1.3(a) -- the in-plane instability limit state.

    AISC 360-16, Sect. H1.3(a), p. 16.1-79: Eqs. H1-1a and H1-1b are used with
    ``Pc`` taken as the available compressive strength **in the plane of
    bending** and ``Mcx`` as the available flexural strength based on the limit
    state of **yielding** -- not the lateral-torsional buckling strength.

    Using the LTB-reduced ``Mcx`` here would double-count out-of-plane
    behaviour, which Sect. H1.3(b) handles separately.
    """
    result = combined_axial_flexure(Pr, Pcx, Mrx=Mrx, Mcx=Mcx_yielding)
    return InteractionResult(
        ratio=result.ratio, citation=result.citation,
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE, terms=result.terms,
        note=(
            "Sect. H1.3(a) in-plane instability: Pc is the in-plane compressive "
            "strength and Mcx is based on YIELDING, not LTB"
        ),
    )


def out_of_plane_buckling(
    Pr: Kip, Pcy: Kip, Mrx: KipIn, Mcx_ltb: KipIn, Cb: float
) -> InteractionResult:
    """Sect. H1.3(b) -- out-of-plane buckling and lateral-torsional buckling.

    AISC 360-16, Eq. H1-3, Sect. H1.3(b), p. 16.1-80::

        (Pr/Pcy)*(1.5 - 0.5*Pr/Pcy) + (Mrx/(Cb*Mcx))^2 <= 1.0

    Parameters
    ----------
    Pcy:
        Available compressive strength **out of** the plane of bending.
    Mcx_ltb:
        Available lateral-torsional strength for major-axis flexure, computed
        with **Cb = 1.0** -- Sect. H1.3(b) says so explicitly, because ``Cb``
        appears separately in the denominator.
    Cb:
        The lateral-torsional buckling modification factor from Sect. F1.

    Notes
    -----
    The User Note on p. 16.1-80 records that ``Cb*Mcx`` may legitimately exceed
    ``phi_b*Mpx``: this equation covers stability, and yielding is caught by
    Eqs. H1-1a and H1-1b. So no cap is applied to the product here -- capping it would make
    the check conservative in a way the Specification does not intend.
    """
    if Pcy <= 0.0 or Mcx_ltb <= 0.0:
        raise GeometryError(f"Pcy and Mcx must be positive, got {Pcy}, {Mcx_ltb}")
    if Cb <= 0.0:
        raise GeometryError(f"Cb must be positive, got {Cb}")

    axial = Pr / Pcy
    axial_term = axial * (1.5 - 0.5 * axial)
    flexure_term = (abs(Mrx) / (Cb * Mcx_ltb)) ** 2
    return InteractionResult(
        ratio=axial_term + flexure_term,
        citation=cite("H1-3"),
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE,
        terms={
            "Pr/Pcy": axial,
            "(Pr/Pcy)(1.5 - 0.5Pr/Pcy)": axial_term,
            "(Mrx/(Cb*Mcx))^2": flexure_term,
        },
        note="Sect. H1.3(b) out-of-plane; Mcx computed with Cb = 1.0",
    )


# ===========================================================================
# Sect. H2 -- unsymmetric and other members
# ===========================================================================
def combined_stress_ratio(
    fra: Ksi, Fca: Ksi, frbw: Ksi = 0.0, Fcbw: Ksi = 0.0, frbz: Ksi = 0.0, Fcbz: Ksi = 0.0
) -> InteractionResult:
    """Combined axial and flexural **stress**, Eq. H2-1.

    AISC 360-16, Sect. H2, p. 16.1-80::

        fra/Fca + frbw/Fcbw + frbz/Fcbz <= 1.0

    ``w`` and ``z`` are the **principal** axes of the unsymmetric section, not
    the geometric ``x`` and ``y``; for a doubly symmetric shape they coincide.

    Sect. H2 is permitted for **any** shape in lieu of Sect. H1, and Sect. H1's
    own User Note says so. It is generally more conservative for shapes Sect. H1
    covers, because it is a linear stress interaction with no ``8/9`` relief.

    Sect. H2 also requires the equation to be evaluated "by considering the
    sense of the flexural stresses at the critical points of the cross section.
    The flexural terms are either added to or subtracted from the axial term as
    applicable." Signs are therefore the caller's: pass the stresses as they act
    at the point being checked, and this sums them as given.
    """
    if Fca <= 0.0:
        raise GeometryError(f"available axial stress must be positive, got {Fca}")

    terms = {"fra/Fca": fra / Fca}
    ratio = fra / Fca
    for label, fr, Fc in (("w", frbw, Fcbw), ("z", frbz, Fcbz)):
        if fr == 0.0:
            continue
        if Fc <= 0.0:
            raise GeometryError(f"frb{label} was supplied with Fcb{label} = {Fc}")
        terms[f"frb{label}/Fcb{label}"] = fr / Fc
        ratio += fr / Fc

    return InteractionResult(
        ratio=abs(ratio), citation=cite("H2-1"),
        limit_state=LimitState.COMBINED_AXIAL_FLEXURE, terms=terms,
        note="principal axes w and z; signs as supplied at the point of consideration",
    )


# ===========================================================================
# Sect. H3.1 -- HSS torsional strength
# ===========================================================================
def round_hss_torsional_constant(D: Inch, t: Inch) -> Inch3:
    """Torsional constant ``C`` for a round HSS.

    AISC 360-16, Sect. H3.1 User Note, p. 16.1-83::

        C = pi*(D - t)^2*t / 2

    Explicitly conservative ("may be conservatively taken as"); use the
    tabulated value from AISC *Manual* Part 1 where one exists.
    """
    if D <= 0.0 or t <= 0.0 or t >= D / 2.0:
        raise GeometryError(f"need 0 < t < D/2, got D={D}, t={t}")
    return math.pi * (D - t) ** 2 * t / 2.0


def rectangular_hss_torsional_constant(B: Inch, H: Inch, t: Inch) -> Inch3:
    """Torsional constant ``C`` for a rectangular HSS.

    AISC 360-16, Sect. H3.1 User Note, p. 16.1-83::

        C = 2*(B - t)*(H - t)*t - 4.5*(4 - pi)*t^3

    The subtracted term removes the corner material that carries no shear flow.
    """
    if min(B, H, t) <= 0.0 or t >= min(B, H) / 2.0:
        raise GeometryError(f"need 0 < t < min(B,H)/2, got B={B}, H={H}, t={t}")
    return 2.0 * (B - t) * (H - t) * t - 4.5 * (4.0 - math.pi) * t**3


def round_hss_torsional_stress(D: Inch, t: Inch, L: Inch, E: Ksi, Fy: Ksi) -> tuple[Ksi, str]:
    """Critical torsional stress for a round HSS.

    AISC 360-16, Eqs. H3-2a and H3-2b, Sect. H3.1(a), p. 16.1-82. ``Fcr`` is
    the **larger** of::

        Fcr = 1.23*E / (sqrt(L/D) * (D/t)^(5/4))                    (Eq. H3-2a)
        Fcr = 0.60*E / (D/t)^(3/2)                                  (Eq. H3-2b)

    "but shall not exceed 0.6*Fy".

    Structurally identical to Eqs. G5-2a/G5-2b for round-HSS *shear*, but with
    different coefficients -- 1.23 and 0.60 here against 1.60 and 0.78 there,
    both about 77% of the shear values. Substituting one pair for the other is
    a plausible-looking error that this pairing makes easy to catch.

    Note ``L`` is the **length of the member**, where Eq. G5-2a uses ``Lv``, the
    distance from maximum to zero shear.
    """
    if min(D, t, L) <= 0.0:
        raise GeometryError(f"D, t and L must be positive, got {D}, {t}, {L}")

    D_over_t = D / t
    a = 1.23 * E / (math.sqrt(L / D) * D_over_t**1.25)  # Eq. H3-2a
    b = 0.60 * E / D_over_t**1.5  # Eq. H3-2b
    yielding = 0.6 * Fy

    buckling = max(a, b)
    if buckling >= yielding:
        return yielding, "torsional yielding (0.6*Fy governs)"
    return buckling, f"Eq. {'H3-2a' if a >= b else 'H3-2b'}"


def rectangular_hss_torsional_stress(h_over_t: Ratio, E: Ksi, Fy: Ksi) -> tuple[Ksi, str]:
    """Critical torsional stress for a rectangular HSS.

    AISC 360-16, Eqs. H3-3, H3-4 and H3-5, Sect. H3.1(b), p. 16.1-82::

        h/t <= 2.45*sqrt(E/Fy):                Fcr = 0.6*Fy            (H3-3)
        2.45*sqrt(E/Fy) < h/t <= 3.07*sqrt(E/Fy):
            Fcr = 0.6*Fy*(2.45*sqrt(E/Fy))/(h/t)                       (H3-4)
        3.07*sqrt(E/Fy) < h/t <= 260:
            Fcr = 0.458*pi^2*E/(h/t)^2                                 (H3-5)

    ``h`` is the flat width of the **longer** side, per Sect. B4.1b(d).

    Raises
    ------
    OutOfScopeError
        If ``h/t > 260``. Sect. H3.1(b)(3) stops there and gives nothing beyond.
    """
    if h_over_t <= 0.0:
        raise GeometryError(f"h/t must be positive, got {h_over_t}")

    root = math.sqrt(E / Fy)
    if h_over_t <= 2.45 * root:
        return 0.6 * Fy, "Eq. H3-3, torsional yielding"
    if h_over_t <= 3.07 * root:
        return 0.6 * Fy * (2.45 * root) / h_over_t, "Eq. H3-4, inelastic buckling"
    if h_over_t <= 260.0:
        return 0.458 * math.pi**2 * E / h_over_t**2, "Eq. H3-5, elastic buckling"
    raise OutOfScopeError(
        f"h/t = {h_over_t:.1f} exceeds 260. AISC 360-16 Sect. H3.1(b), p. 16.1-82, "
        "provides no torsional strength for a rectangular HSS beyond that limit."
    )


def hss_torsional_strength(
    C: Inch3, Fcr: Ksi, governing: str, *, basis: Basis = Basis.LRFD
) -> StrengthResult:
    """Available torsional strength of an HSS.

    AISC 360-16, Eq. H3-1, Sect. H3.1, p. 16.1-81::

        Tn = Fcr*C

    with ``phi_T = 0.90`` (LRFD) and ``Omega_T = 1.67`` (ASD).
    """
    if C <= 0.0 or Fcr <= 0.0:
        raise GeometryError(f"C and Fcr must be positive, got {C}, {Fcr}")
    state = LimitStateResult(
        LimitState.TORSIONAL_YIELDING if "yielding" in governing
        else LimitState.TORSIONAL_BUCKLING_HSS,
        Fcr * C, cite("H3-1"), detail={"Fcr": Fcr, "C": C}, note=governing,
    )
    return StrengthResult.build(
        "Tn", "kip-in.", [state], phi=PHI_T_TORSION, omega=OMEGA_T_TORSION, basis=basis
    )


# ===========================================================================
# Sect. H3.2 -- HSS under combined torsion, shear, flexure and axial force
# ===========================================================================
def torsion_may_be_neglected(Tr: KipIn, Tc: KipIn) -> bool:
    """Whether Sect. H3.2 permits torsion to be ignored.

    AISC 360-16, Sect. H3.2, p. 16.1-83: "When the required torsional strength,
    Tr, is less than or equal to 20% of the available torsional strength, Tc,
    the interaction of torsion, shear, flexure and/or axial force for HSS may be
    determined by Section H1 and the torsional effects may be neglected."
    """
    if Tc <= 0.0:
        raise GeometryError(f"available torsional strength must be positive, got {Tc}")
    return abs(Tr) / Tc <= TORSION_NEGLECT_THRESHOLD


def hss_combined_torsion(
    Pr: Kip, Pc: Kip, Mr: KipIn, Mc: KipIn, Vr: Kip, Vc: Kip, Tr: KipIn, Tc: KipIn
) -> InteractionResult:
    """Combined torsion, shear, flexure and axial force in an HSS.

    AISC 360-16, Eq. H3-6, Sect. H3.2, p. 16.1-83::

        (Pr/Pc + Mr/Mc) + (Vr/Vc + Tr/Tc)^2 <= 1.0

    Note the grouping: the axial and flexural terms are **linear**, the shear
    and torsional terms are **squared as a pair**. Squaring them individually,
    or squaring the whole expression, both give the wrong answer -- the second
    bracket is squared after summing, because shear and torsion are both shear
    flows that add directly before interacting quadratically with normal stress.

    Applies only when ``Tr > 0.20*Tc``; below that
    :func:`torsion_may_be_neglected` permits Sect. H1 instead.
    """
    for name, value in (("Pc", Pc), ("Mc", Mc), ("Vc", Vc), ("Tc", Tc)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")

    axial_flexure = abs(Pr) / Pc + abs(Mr) / Mc
    shear_torsion = abs(Vr) / Vc + abs(Tr) / Tc
    return InteractionResult(
        ratio=axial_flexure + shear_torsion**2,
        citation=cite("H3-6"),
        limit_state=LimitState.COMBINED_TORSION,
        terms={
            "Pr/Pc": abs(Pr) / Pc,
            "Mr/Mc": abs(Mr) / Mc,
            "Vr/Vc": abs(Vr) / Vc,
            "Tr/Tc": abs(Tr) / Tc,
            "(Vr/Vc + Tr/Tc)^2": shear_torsion**2,
        },
        note="shear and torsion sum first, then square -- Eq. H3-6",
    )


# ===========================================================================
# Sect. H3.3 -- non-HSS members in torsion
# ===========================================================================
def non_hss_torsional_stress(Fy: Ksi, *, Fcr: Ksi | None = None) -> StrengthResult:
    """Available torsional **stress** for a non-HSS member.

    AISC 360-16, Sect. H3.3, p. 16.1-84, with ``phi_T = 0.90`` / ``Omega_T = 1.67``::

        yielding under normal stress:  Fn = Fy          (Eq. H3-7)
        shear yielding under shear:    Fn = 0.6*Fy      (Eq. H3-8)
        buckling:                      Fn = Fcr         (Eq. H3-9)

    Sect. H3.3 asks for "the lowest value obtained" among the three. ``Fcr`` is
    "as determined by analysis" -- the Specification gives no formula, so it is
    an input; omit it and only the two yielding states are evaluated, with the
    result saying so.

    Sect. H3.3 closes with "Constrained local yielding is permitted adjacent to
    areas that remain elastic", which is why Eq. H3-7 uses the full ``Fy``
    rather than a reduced value.
    """
    if Fy <= 0.0:
        raise GeometryError(f"Fy must be positive, got {Fy}")

    states = [
        LimitStateResult(
            LimitState.TORSIONAL_YIELDING, Fy, cite("H3-7"),
            note="yielding under normal stress",
        ),
        LimitStateResult(
            LimitState.SHEAR_YIELDING, 0.6 * Fy, cite("H3-8"),
            note="shear yielding under shear stress",
        ),
    ]
    if Fcr is not None:
        states.append(
            LimitStateResult(
                LimitState.TORSIONAL_BUCKLING_HSS, Fcr, cite("H3-9"),
                note="buckling; Fcr determined by analysis",
            )
        )
    else:
        states[0] = LimitStateResult(
            states[0].limit_state, states[0].nominal, states[0].citation,
            note=(
                "yielding under normal stress; INCOMPLETE -- Sect. H3.3 also "
                "requires a buckling check, with Fcr determined by analysis"
            ),
        )
    return StrengthResult.build(
        "Fn", "ksi", states, phi=PHI_T_TORSION, omega=OMEGA_T_TORSION, basis=Basis.LRFD
    )


# ===========================================================================
# Sect. H4 -- rupture of flanges with holes
# ===========================================================================
def flange_rupture_interaction(Pr: Kip, Pc: Kip, Mrx: KipIn, Mcx: KipIn) -> InteractionResult:
    """Tensile rupture of a bolted flange under combined axial force and flexure.

    AISC 360-16, Eq. H4-1, Sect. H4, p. 16.1-84::

        Pr/Pc + Mrx/Mcx <= 1.0

    Checked **at the bolt holes**, and "each flange subjected to tension due to
    axial force and flexure shall be checked separately".

    The sign convention is unusual and load-bearing: ``Pr`` is **positive in
    tension and negative in compression**, and ``Mrx`` is **positive for tension
    in the flange under consideration**. A compressive axial force therefore
    *reduces* the ratio, which is physically right -- it relieves the tension the
    holes have to carry. Taking absolute values, as the other interaction
    equations do, would be wrong here, so the signs are passed straight through.

    ``Pc`` is the tensile-rupture strength from Sect. D2(b) (phi = 0.75), and
    ``Mcx`` is from Sect. F13.1, or ``Mp`` computed ignoring the holes where the
    F13.1 rupture limit state does not apply.
    """
    if Pc <= 0.0 or Mcx <= 0.0:
        raise GeometryError(f"Pc and Mcx must be positive, got {Pc}, {Mcx}")

    axial, flexure = Pr / Pc, Mrx / Mcx
    return InteractionResult(
        ratio=max(axial + flexure, 0.0),
        citation=cite("H4-1"),
        limit_state=LimitState.FLANGE_RUPTURE,
        terms={"Pr/Pc": axial, "Mrx/Mcx": flexure},
        note=(
            "signed: Pr positive in tension, Mrx positive for tension in this "
            "flange; check each flange separately"
        ),
    )
