"""Appendix 4 -- Structural Design for Fire Conditions.

Covers Sects. 4.1 through 4.3, pp. 16.1-222 to 16.1-233.

Fire design replaces material properties, not equations: Sect. 4.2.4d routes
every member check back through Chapters E, F, G, H and I with ``Fy(T)``,
``Fu(T)`` and ``E(T)`` from Table A-4.2.1 in place of the ambient values. Only
two provisions are new -- Eq. A-4-2 for compression and Eqs. A-4-3 through
A-4-10 for lateral-torsional buckling -- and both exist because the ambient
curves are calibrated on residual-stress patterns that no longer apply once the
steel is hot.

Three things govern how this module is built:

**Below 400 F, nothing changes.** Sect. 4.2.4d: "For steel temperatures less
than or equal to 400 F (200 C), the member and connection design strengths shall
be determined without consideration of temperature effects." That is a hard
cut-off, not a smooth taper, and it is why Table A-4.2.1 marks ``ky`` and ``ku``
with an asterisk below 750 F meaning "use ambient properties".

**LRFD only.** Sect. 4.1.2: "Structural design for fire conditions using
Appendix 4.2 shall be performed using the load and resistance factor design
method". There is no ASD path, so nothing here takes a ``Basis``.

**The tables are interpolated, not stepped.** They tabulate discrete
temperatures; a member at 1100 F needs a value between the 1000 F and 1200 F
rows, and linear interpolation is the standard reading.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

from .core.exceptions import GeometryError, OutOfScopeError
from .core.units import Inch, Inch3, Inch4, KipIn, Ksi, Ratio

__all__ = [
    "AMBIENT_TEMPERATURE_F",
    "NO_REDUCTION_TEMPERATURE_F",
    "TABLE_A4_2_1",
    "TABLE_A4_2_2",
    "TABLE_A4_2_3",
    "SteelAtTemperature",
    "steel_properties",
    "concrete_properties",
    "bolt_retention",
    "fire_load_combination",
    "Fcr_at_temperature",
    "FL_at_temperature",
    "Mp_at_temperature",
    "Mr_at_temperature",
    "cx_coefficient",
    "Lr_at_temperature",
    "Fcr_ltb_at_temperature",
    "Mn_ltb_at_temperature",
    "composite_flexural_at_temperature",
]

#: Ambient reference temperature, degrees F.
AMBIENT_TEMPERATURE_F: float = 68.0

#: Sect. 4.2.4d, p. 16.1-229: at or below this, temperature effects are ignored.
NO_REDUCTION_TEMPERATURE_F: float = 400.0

#: Table A-4.2.1, p. 16.1-226 -- ``T (deg F) -> (kE, kp, ky, ku)`` where
#: ``kE = E(T)/E = G(T)/G``, ``kp = Fp(T)/Fy``, ``ky = Fy(T)/Fy`` and
#: ``ku = Fu(T)/Fy``.
#:
#: ``ky`` and ``ku`` are marked "*use ambient properties*" below 750 F, which is
#: recorded here as 1.00 -- the asterisk means the reduction has not begun, not
#: that the value is undefined.
TABLE_A4_2_1: dict[float, tuple[float, float, float, float]] = {
    68.0: (1.00, 1.00, 1.00, 1.00),
    200.0: (1.00, 1.00, 1.00, 1.00),
    400.0: (0.90, 0.80, 1.00, 1.00),
    600.0: (0.78, 0.58, 1.00, 1.00),
    750.0: (0.70, 0.42, 1.00, 1.00),
    800.0: (0.67, 0.40, 0.94, 0.94),
    1000.0: (0.49, 0.29, 0.66, 0.66),
    1200.0: (0.22, 0.13, 0.35, 0.35),
    1400.0: (0.11, 0.06, 0.16, 0.16),
    1600.0: (0.07, 0.04, 0.07, 0.07),
    1800.0: (0.05, 0.03, 0.04, 0.04),
    2000.0: (0.02, 0.01, 0.02, 0.02),
    2200.0: (0.00, 0.00, 0.00, 0.00),
}

#: Table A-4.2.2, p. 16.1-227 -- ``T (deg F) -> (kc normal weight,
#: kc lightweight, Ec(T)/Ec, epsilon_cu percent)``.
TABLE_A4_2_2: dict[float, tuple[float, float, float, float]] = {
    68.0: (1.00, 1.00, 1.00, 0.25),
    200.0: (0.95, 1.00, 0.93, 0.34),
    400.0: (0.90, 1.00, 0.75, 0.46),
    550.0: (0.86, 1.00, 0.61, 0.58),
    600.0: (0.83, 0.98, 0.57, 0.62),
    800.0: (0.71, 0.85, 0.38, 0.80),
    1000.0: (0.54, 0.71, 0.20, 1.06),
    1200.0: (0.38, 0.58, 0.092, 1.32),
    1400.0: (0.21, 0.45, 0.073, 1.43),
    1600.0: (0.10, 0.31, 0.055, 1.49),
    1800.0: (0.05, 0.18, 0.036, 1.50),
    2000.0: (0.01, 0.05, 0.018, 1.50),
    2200.0: (0.00, 0.00, 0.000, 0.00),
}

#: Table A-4.2.3, p. 16.1-228 -- ``T (deg F) -> Fnt(T)/Fnt = Fnv(T)/Fnv`` for
#: Group A and Group B high-strength bolts.
TABLE_A4_2_3: dict[float, float] = {
    68.0: 1.00,
    200.0: 0.97,
    300.0: 0.95,
    400.0: 0.93,
    600.0: 0.88,
    800.0: 0.71,
    900.0: 0.59,
    1000.0: 0.42,
    1200.0: 0.16,
    1400.0: 0.08,
    1600.0: 0.04,
    1800.0: 0.01,
    2000.0: 0.00,
}


def _interpolate(
    table: Mapping[float, tuple[float, ...] | float],
    T: float,
    index: int | None = None,
) -> float:
    """Linearly interpolate a Table A-4.2.x column at temperature ``T``.

    The tables give discrete temperatures; a member at 1100 F sits between rows
    and linear interpolation is the standard reading. Extrapolating past 2200 F
    is refused -- the tables end at zero and there is nothing beyond.
    """
    temperatures = sorted(table)
    if temperatures[0] > T or temperatures[-1] < T:
        raise OutOfScopeError(
            f"temperature {T:g} F is outside the tabulated range "
            f"{temperatures[0]:g}-{temperatures[-1]:g} F of Appendix 4"
        )

    def value_at(t: float) -> float:
        row = table[t]
        if index is not None and isinstance(row, tuple):
            return row[index]
        assert not isinstance(row, tuple)
        return float(row)

    if T in table:
        return value_at(T)
    upper = next(t for t in temperatures if t > T)
    lower = max(t for t in temperatures if t < T)
    fraction = (T - lower) / (upper - lower)
    return value_at(lower) + (value_at(upper) - value_at(lower)) * fraction


@dataclass(frozen=True, slots=True)
class SteelAtTemperature:
    """Steel properties at an elevated temperature, from Table A-4.2.1."""

    T: float
    kE: float  # noqa: N815 -- Table A-4.2.1's own symbol for the stiffness retention factor
    kp: float
    ky: float
    ku: float
    E: Ksi
    Fy: Ksi
    Fu: Ksi
    Fp: Ksi
    reduced: bool

    def report(self, width: int = 78) -> str:
        lines = ["=" * width, f"Table A-4.2.1 -- steel at {self.T:g} F", "=" * width]
        lines.append(f"{'kE = E(T)/E = G(T)/G':<52}{self.kE:>24,.4f}")
        lines.append(f"{'kp = Fp(T)/Fy':<52}{self.kp:>24,.4f}")
        lines.append(f"{'ky = Fy(T)/Fy':<52}{self.ky:>24,.4f}")
        lines.append(f"{'ku = Fu(T)/Fy':<52}{self.ku:>24,.4f}")
        lines.append("-" * width)
        lines.append(f"{'E(T), ksi':<52}{self.E:>24,.1f}")
        lines.append(f"{'Fy(T), ksi':<52}{self.Fy:>24,.2f}")
        lines.append(f"{'Fu(T), ksi':<52}{self.Fu:>24,.2f}")
        if not self.reduced:
            lines.append("    Sect. 4.2.4d: at or below 400 F, temperature effects are ignored")
        lines.append("=" * width)
        return "\n".join(lines)


def steel_properties(T: float, Fy: Ksi, E: Ksi = 29000.0, Fu: Ksi = 0.0) -> SteelAtTemperature:
    """Steel properties at elevated temperature, Table A-4.2.1, p. 16.1-226.

    Returns the four retention factors and the reduced properties.

    Note ``ku`` is ``Fu(T)/**Fy**``, not ``Fu(T)/Fu`` -- the table normalises the
    elevated tensile strength on the **yield** stress. Reading it as a fraction
    of ``Fu`` overstates ``Fu(T)`` by the ``Fu/Fy`` ratio, about 30% for A992.
    At 800 F, ``ky = ku = 0.94``: the two strengths have converged, because hot
    steel loses its strain-hardening reserve before it loses its yield strength.

    Below 400 F, Sect. 4.2.4d directs that temperature effects be ignored
    entirely, so the ambient values are returned and ``reduced`` is False.
    """
    if Fy <= 0.0:
        raise GeometryError(f"Fy must be positive, got {Fy}")

    if T <= NO_REDUCTION_TEMPERATURE_F:
        return SteelAtTemperature(
            T=T, kE=1.0, kp=1.0, ky=1.0, ku=1.0,
            E=E, Fy=Fy, Fu=Fu or Fy, Fp=Fy, reduced=False,
        )

    kE = _interpolate(TABLE_A4_2_1, T, 0)
    kp = _interpolate(TABLE_A4_2_1, T, 1)
    ky = _interpolate(TABLE_A4_2_1, T, 2)
    ku = _interpolate(TABLE_A4_2_1, T, 3)
    return SteelAtTemperature(
        T=T, kE=kE, kp=kp, ky=ky, ku=ku,
        E=kE * E, Fy=ky * Fy, Fu=ku * Fy, Fp=kp * Fy, reduced=True,
    )


def concrete_properties(T: float, *, lightweight: bool = False) -> tuple[float, float, float]:
    """Concrete properties at elevated temperature, Table A-4.2.2, p. 16.1-227.

    Returns ``(kc, Ec(T)/Ec, epsilon_cu percent)``.

    **Lightweight concrete holds up markedly better** -- ``kc = 1.00`` all the
    way to 550 F where normal weight is already down to 0.86, and 0.71 at
    1000 F against 0.54. The aggregate is already fired, so it does not undergo
    the further mineralogical change that degrades normal-weight aggregate.

    ``Ec(T)/Ec`` is tabulated for normal weight only and falls far faster than
    ``kc``: 0.20 at 1000 F against a strength retention of 0.54. Concrete goes
    soft well before it goes weak, which is what drives fire deflections.
    """
    kc = _interpolate(TABLE_A4_2_2, T, 1 if lightweight else 0)
    kEc = _interpolate(TABLE_A4_2_2, T, 2)
    ecu = _interpolate(TABLE_A4_2_2, T, 3)
    return kc, kEc, ecu


def bolt_retention(T: float) -> float:
    """High-strength bolt strength retention, Table A-4.2.3, p. 16.1-228.

    Returns ``Fnt(T)/Fnt = Fnv(T)/Fnv`` for Group A and Group B bolts -- the
    same factor applies to tension and shear.

    Bolts degrade **earlier** than the steel they connect: 0.93 at 400 F where
    Table A-4.2.1 still gives ``ky = 1.00``, and 0.42 at 1000 F against 0.66.
    A connection can therefore become the weak link in a fire even when the
    members are still adequate.
    """
    return _interpolate(TABLE_A4_2_3, T)


def fire_load_combination(
    D: float, AT: float, L: float, S: float, *, uplift: bool = False
) -> float:
    """Required strength under the design-basis fire.

    AISC 360-16, Eq. A-4-1, Sect. 4.1.4, p. 16.1-223::

        (0.9 or 1.2)*D + AT + 0.5*L + 0.2*S

    ``AT`` is the nominal forces and deformations due to the design-basis fire,
    and enters **unfactored**. The live and snow factors are the extraordinary
    -event values from ASCE/SEI 7 Sect. 2.5, not the ordinary strength ones.

    Use 0.9 on dead load where it acts to relieve -- fire-induced thermal
    expansion can push a member in the same direction as uplift.
    """
    return (0.9 if uplift else 1.2) * D + AT + 0.5 * L + 0.2 * S


def Fcr_at_temperature(Fy_T: Ksi, Fe_T: Ksi) -> Ksi:
    """Critical stress for flexural buckling at elevated temperature.

    AISC 360-16, Eq. A-4-2, Sect. 4.2.4d(b), p. 16.1-229::

        Fcr(T) = [0.42^sqrt(Fy(T)/Fe(T))]*Fy(T)

    Used **in lieu of Eqs. E3-2 and E3-3**, with ``Fe(T)`` from Eq. E3-4
    evaluated at ``E(T)``.

    Two differences from Eq. E3-2 matter. The base is **0.42, not 0.658**, and
    the exponent carries a **square root** that Eq. E3-2 does not have. Together
    they give a much flatter curve: hot steel has no distinct
    elastic/inelastic transition, because the residual stresses that create one
    at ambient temperature have relaxed away. That is also why there is no
    second branch -- Eq. A-4-2 covers the whole slenderness range on its own.
    """
    if Fy_T <= 0.0 or Fe_T <= 0.0:
        raise GeometryError(f"Fy(T) and Fe(T) must be positive, got {Fy_T}, {Fe_T}")
    return math.pow(0.42, math.sqrt(Fy_T / Fe_T)) * Fy_T


def FL_at_temperature(Fy: Ksi, kp: float, ky: float) -> Ksi:
    """Nominal compression flange stress at elevated temperature.

    AISC 360-16, Eq. A-4-8, Sect. 4.2.4d(c), p. 16.1-230::

        FL(T) = Fy*(kp - 0.3*ky)

    Note it is built from the **ambient** ``Fy`` times a combination of two
    retention factors -- not from ``Fy(T)``. At ambient both factors are 1.0 and
    the expression gives ``0.7*Fy``, exactly the ``FL`` of Eq. F4-6a. The fire
    version generalises it: as ``kp`` falls faster than ``ky``, the proportional
    limit drops relative to yield and the inelastic LTB range widens.
    """
    if Fy <= 0.0:
        raise GeometryError(f"Fy must be positive, got {Fy}")
    return Fy * (kp - 0.3 * ky)


def Mp_at_temperature(Fy_T: Ksi, Zx: Inch3) -> KipIn:
    """Plastic moment at elevated temperature.

    AISC 360-16, Eq. A-4-9, Sect. 4.2.4d(c), p. 16.1-230::

        Mp(T) = Fy(T)*Zx
    """
    if Fy_T <= 0.0 or Zx <= 0.0:
        raise GeometryError(f"Fy(T) and Zx must be positive, got {Fy_T}, {Zx}")
    return Fy_T * Zx


def Mr_at_temperature(FL_T: Ksi, Sx: Inch3) -> KipIn:
    """Limiting buckling moment at elevated temperature.

    AISC 360-16, Eq. A-4-7, Sect. 4.2.4d(c), p. 16.1-230::

        Mr(T) = FL(T)*Sx
    """
    if FL_T <= 0.0 or Sx <= 0.0:
        raise GeometryError(f"FL(T) and Sx must be positive, got {FL_T}, {Sx}")
    return FL_T * Sx


def cx_coefficient(T: float) -> float:
    """Exponent for the elevated-temperature LTB curve.

    AISC 360-16, Eq. A-4-10, Sect. 4.2.4d(c), p. 16.1-230::

        cx = 0.53 + T/450  <= 3.0,   T in degrees F
        cx = 0.6  + T/250  <= 3.0,   T in degrees C  (Eq. A-4-10M)

    This is what replaces Eq. F2-2's straight line: the inelastic LTB curve is
    raised to the power ``cx``, which grows from 0.68 at 68 F to the 3.0 cap at
    1112 F. A hot beam loses strength far more sharply with unbraced length than
    a cold one, so the transition bows downward instead of running straight.

    The two forms are **not** unit conversions of each other: at 400 F (204 C)
    the Fahrenheit form gives 1.42 and the Celsius form 1.42 -- they agree, but
    only because both constants were fitted, not converted.
    """
    return min(0.53 + T / 450.0, 3.0)


def Lr_at_temperature(
    rts: Inch, E_T: Ksi, FL_T: Ksi, J: Inch4, c: Ratio, Sx: Inch3, ho: Inch
) -> Inch:
    """Limiting unbraced length at elevated temperature.

    AISC 360-16, Eq. A-4-6, Sect. 4.2.4d(c), p. 16.1-230::

        Lr(T) = 1.95*rts*(E(T)/FL(T))
                *sqrt( J*c/(Sx*ho)
                       + sqrt( (J*c/(Sx*ho))^2 + 6.76*(FL(T)/E(T))^2 ) )

    Eq. F2-6 with ``E(T)`` for ``E`` and ``FL(T)`` for ``0.7*Fy``. The same
    rounded 1.95 and 6.76 apply, with the same 0.12% branch mismatch documented
    for Eq. F2-6.
    """
    for name, value in (("rts", rts), ("Sx", Sx), ("ho", ho), ("c", c)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    if E_T <= 0.0 or FL_T <= 0.0:
        raise GeometryError(f"E(T) and FL(T) must be positive, got {E_T}, {FL_T}")

    term = J * c / (Sx * ho)
    return 1.95 * rts * (E_T / FL_T) * math.sqrt(
        term + math.sqrt(term**2 + 6.76 * (FL_T / E_T) ** 2)
    )


def Fcr_ltb_at_temperature(
    Cb: float, Lb: Inch, rts: Inch, E_T: Ksi, J: Inch4, c: Ratio, Sx: Inch3, ho: Inch
) -> Ksi:
    """Elastic LTB stress at elevated temperature.

    AISC 360-16, Eq. A-4-5, Sect. 4.2.4d(c), p. 16.1-230::

        Fcr(T) = Cb*pi^2*E(T)/(Lb/rts)^2
                 *sqrt(1 + 0.078*J*c/(Sx*ho)*(Lb/rts)^2)

    Eq. F2-4 with ``E(T)`` in place of ``E`` -- structurally unchanged.
    """
    if Lb <= 0.0 or rts <= 0.0:
        raise GeometryError(f"Lb and rts must be positive, got {Lb}, {rts}")
    slenderness = Lb / rts
    return (
        Cb * math.pi**2 * E_T / slenderness**2
        * math.sqrt(1.0 + 0.078 * J * c / (Sx * ho) * slenderness**2)
    )


def Mn_ltb_at_temperature(
    Lb: Inch, Lr_T: Inch, Mp_T: KipIn, Mr_T: KipIn, cx: float,
    *, Cb: float = 1.0, Fcr_T: Ksi = 0.0, Sx: Inch3 = 0.0,
) -> tuple[KipIn, str]:
    """Nominal LTB strength at elevated temperature.

    AISC 360-16, Eqs. A-4-3 and A-4-4, Sect. 4.2.4d(c), p. 16.1-230::

        Lb <= Lr(T):  Mn(T) = Cb*[Mp(T) - (Mp(T) - Mr(T))*(Lb/Lr(T))^cx]
                              <= Mp(T)                             (A-4-3)
        Lb >  Lr(T):  Mn(T) = Fcr(T)*Sx  <= Mp(T)                  (A-4-4)

    Used **in lieu of Eqs. F2-2 through F2-6**.

    Two structural differences from Eq. F2-2. There is **no Lp** -- the curve
    starts falling immediately from ``Lb = 0``, because a hot beam has no
    plastic plateau. And the interpolation variable is ``Lb/Lr(T)`` raised to
    ``cx``, not ``(Lb - Lp)/(Lr - Lp)`` linear. At ``Lb = Lr(T)`` the bracket
    gives ``Mp - (Mp - Mr) = Mr`` exactly, so the two branches meet where
    Eq. A-4-4's ``Fcr(T)*Sx`` equals ``Mr(T)`` by construction.
    """
    if Lr_T <= 0.0 or Mp_T <= 0.0:
        raise GeometryError(f"Lr(T) and Mp(T) must be positive, got {Lr_T}, {Mp_T}")

    if Lb <= Lr_T:
        bracket = Mp_T - (Mp_T - Mr_T) * math.pow(Lb / Lr_T, cx)
        return (
            min(Cb * bracket, Mp_T),
            f"Eq. A-4-3: Lb <= Lr(T), cx = {cx:.3f} -- no Lp plateau at temperature",
        )
    if Sx <= 0.0 or Fcr_T <= 0.0:
        raise GeometryError("the Lb > Lr(T) branch needs Fcr(T) and Sx")
    return min(Fcr_T * Sx, Mp_T), "Eq. A-4-4: Lb > Lr(T), elastic LTB"


def composite_flexural_at_temperature(r_T: float, Mn: KipIn) -> KipIn:
    """Nominal flexural strength of a composite beam at elevated temperature.

    AISC 360-16, Eq. A-4-11, Sect. 4.2.4d(d), p. 16.1-231::

        Mn(T) = r(T)*Mn

    ``Mn`` is the **ambient** Chapter I strength and ``r(T)`` is the retention
    factor from Table A-4.2.4, indexed on the **bottom flange** temperature.

    Sect. 4.2.4d(c) permits the bottom flange temperature to be assumed constant
    over the depth for steel beams; for a composite beam the slab keeps the top
    much cooler, so the bottom flange is both the hottest point and the one
    carrying the tension -- which is why the whole member is indexed on it.
    """
    if not 0.0 <= r_T <= 1.0:
        raise GeometryError(f"the retention factor must lie in [0, 1], got {r_T}")
    if Mn <= 0.0:
        raise GeometryError(f"Mn must be positive, got {Mn}")
    return r_T * Mn
