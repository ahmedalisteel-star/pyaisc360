"""Appendix 2 -- Design for Ponding.

Covers Sects. 2.1 and 2.2, pp. 16.1-192 to 16.1-195.

Ponding is a *stability* problem wearing a serviceability disguise: rainwater
collects in a roof's deflection, the added weight deepens the deflection, and
the two can diverge. Appendix 2 offers a simplified stiffness check (Sect. 2.1)
and a stress-based alternative (Sect. 2.2) when the simplified one fails.

Two things about Eqs. A-2-3 and A-2-4 are easy to get wrong and invisible once
wrong:

**They are dimensional.** Lengths go in **feet**, moments of inertia in in.^4,
and the ``10^7`` divisor reconciles the two. Passing lengths in inches
understates ``Cp`` by 20736 -- a factor of ``12^4`` -- and the check passes
trivially.

**A steel deck is a secondary member**, per Sect. 2.1, "when it is directly
supported by the primary members". So a deck-on-joists-on-girders roof and a
deck-on-girders roof put different things in ``Cs``.
"""

from __future__ import annotations

from dataclasses import dataclass

from .core.exceptions import GeometryError
from .core.units import Inch4, Ksi, Ratio

__all__ = [
    "PONDING_LIMIT",
    "DECK_STIFFNESS_COEFFICIENT",
    "PondingCheck",
    "Cp_primary",
    "Cs_secondary",
    "simplified_ponding_check",
    "minimum_deck_inertia",
    "stress_index",
]

#: Eq. A-2-1, p. 16.1-192 -- the combined flexibility limit.
PONDING_LIMIT: Ratio = 0.25

#: Eq. A-2-2 -- minimum deck moment of inertia coefficient, in.^4 per ft.
DECK_STIFFNESS_COEFFICIENT: float = 25.0e-6


def Cp_primary(Ls: float, Lp: float, Ip: Inch4) -> Ratio:
    """Flexibility coefficient of the primary members.

    AISC 360-16, Eq. A-2-3, Sect. 2.1, p. 16.1-192::

        Cp = 32*Ls*Lp^4/(10^7*Ip)

    ``Ls`` and ``Lp`` are in **feet**; ``Ip`` is in in.^4. The ``10^7`` divisor
    carries the unit reconciliation, so the expression is dimensional and not a
    general formula. The metric form, Eq. A-2-3M, uses ``504*Ls*Lp^4/Ip`` with
    metres and mm^4 -- a different constant, not a converted one.

    ``Lp^4`` means primary-member length dominates: doubling a girder span makes
    the roof sixteen times more ponding-flexible.
    """
    for name, value in (("Ls", Ls), ("Lp", Lp), ("Ip", Ip)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    return 32.0 * Ls * Lp**4 / (1.0e7 * Ip)


def Cs_secondary(S: float, Ls: float, Is: Inch4) -> Ratio:
    """Flexibility coefficient of the secondary members.

    AISC 360-16, Eq. A-2-4, Sect. 2.1, p. 16.1-192::

        Cs = 32*S*Ls^4/(10^7*Is)

    ``S`` is the secondary-member spacing and ``Ls`` their length, both in feet.

    Same form as Eq. A-2-3 with the spacing in place of the other member's
    length -- so a roof framed with widely spaced, long-span joists is the worst
    case on both counts.
    """
    for name, value in (("S", S), ("Ls", Ls), ("Is", Is)):
        if value <= 0.0:
            raise GeometryError(f"{name} must be positive, got {value}")
    return 32.0 * S * Ls**4 / (1.0e7 * Is)


def minimum_deck_inertia(S: float) -> Inch4:
    """Minimum deck moment of inertia for the simplified check.

    AISC 360-16, Eq. A-2-2, Sect. 2.1, p. 16.1-192::

        Id >= 25*(S^4)*10^-6,  in.^4 per ft, with S in feet

    A separate requirement from Eq. A-2-1, not part of it -- both must hold.
    """
    if S <= 0.0:
        raise GeometryError(f"S must be positive, got {S}")
    return DECK_STIFFNESS_COEFFICIENT * S**4


@dataclass(frozen=True, slots=True)
class PondingCheck:
    """Result of the Sect. 2.1 simplified ponding check."""

    stable: bool
    Cp: Ratio
    Cs: Ratio
    combined: Ratio
    deck_adequate: bool
    Id_required: Inch4
    note: str

    def report(self, width: int = 78) -> str:
        lines = ["=" * width, "Appendix 2.1 -- simplified design for ponding", "=" * width]
        lines.append(f"{'Cp  (Eq. A-2-3)':<52}{self.Cp:>24,.4f}")
        lines.append(f"{'Cs  (Eq. A-2-4)':<52}{self.Cs:>24,.4f}")
        lines.append(f"{'Cp + 0.9*Cs  (Eq. A-2-1, limit 0.25)':<52}{self.combined:>24,.4f}")
        lines.append(f"{'Id required, in.^4/ft  (Eq. A-2-2)':<52}{self.Id_required:>24,.4f}")
        lines.append("-" * width)
        lines.append(f"    {self.note}")
        lines.append("=" * width)
        return "\n".join(lines)


def simplified_ponding_check(
    Ls: float, Lp: float, S: float, Ip: Inch4, Is: Inch4, Id: Inch4
) -> PondingCheck:
    """The Sect. 2.1 simplified design for ponding, p. 16.1-192.

    Stable if **both** hold::

        Cp + 0.9*Cs <= 0.25                                       (A-2-1)
        Id >= 25*(S^4)*10^-6                                      (A-2-2)

    All lengths in **feet**, all moments of inertia in in.^4.

    The ``0.9`` weighting on ``Cs`` is not a rounding -- secondary members
    deflect within the primary members' deflection, so their contribution to the
    total ponding depth is slightly less than proportional.

    Sect. 2.1 adds that for trusses and steel joists, ``Ip`` and ``Is`` "shall
    include the effects of web member strain", and the User Note gives 15% as a
    typical reduction when the inertia is computed from chord areas alone. That
    correction is the caller's -- it depends on the joist, not on this equation.

    Failing this check is not a failure of the roof: Sect. 2.2 offers a
    stress-based alternative, and the note says so.
    """
    Cp = Cp_primary(Ls, Lp, Ip)
    Cs = Cs_secondary(S, Ls, Is)
    combined = Cp + 0.9 * Cs
    Id_required = minimum_deck_inertia(S)
    deck_adequate = Id >= Id_required

    stiffness_ok = combined <= PONDING_LIMIT
    stable = stiffness_ok and deck_adequate

    if stable:
        note = "stable for ponding; no further investigation needed (Sect. 2.1)"
    else:
        failures = []
        if not stiffness_ok:
            failures.append(f"Cp + 0.9*Cs = {combined:.4f} exceeds {PONDING_LIMIT}")
        if not deck_adequate:
            failures.append(f"Id = {Id:.4f} is below {Id_required:.4f} in.^4/ft")
        note = (
            "; ".join(failures)
            + " -- Sect. 2.2 offers a stress-based alternative before the roof is "
            "condemned; lengths here are in FEET and inertias in in.^4"
        )

    return PondingCheck(
        stable=stable, Cp=Cp, Cs=Cs, combined=combined,
        deck_adequate=deck_adequate, Id_required=Id_required, note=note,
    )


def stress_index(Fy: Ksi, fo: Ksi) -> Ratio:
    """Stress index for the improved ponding design.

    AISC 360-16, Eqs. A-2-5 (primary) and A-2-6 (secondary), Sect. 2.2,
    p. 16.1-193::

        U = (0.8*Fy - fo)/fo

    ``fo`` is the stress from impounded water under nominal rain or snow,
    *excluding* the ponding contribution, plus other concurrent loads. The
    index measures how much reserve stress the member has to absorb ponding.

    The two equations are identical in form; the Specification numbers them
    separately because they take the primary and secondary members' own ``fo``.

    Raises
    ------
    GeometryError
        If ``fo >= 0.8*Fy``. The member has no reserve at all, so the ponding
        contribution cannot be accommodated by any amount of stiffness and the
        Sect. 2.2 charts do not extend there.
    """
    if Fy <= 0.0:
        raise GeometryError(f"Fy must be positive, got {Fy}")
    if fo <= 0.0:
        raise GeometryError(f"fo must be positive, got {fo}")
    if fo >= 0.8 * Fy:
        raise GeometryError(
            f"fo = {fo:g} ksi reaches 0.8*Fy = {0.8 * Fy:g} ksi: the member has no "
            "reserve stress for the ponding contribution and Sect. 2.2 does not apply"
        )
    return (0.8 * Fy - fo) / fo
