"""Result objects: nominal strength, every limit state, and which one governed.

The Specification almost never asks for "the" strength -- it asks for the lowest
value obtained from a list of applicable limit states (Sects. E1, F1, D2, ...).
Returning a bare float throws away the two things an engineer actually needs at
review time: *which* limit state governed, and by how much it beat the others.

So every public strength function returns a :class:`StrengthResult` holding all
limit states it evaluated.  It behaves like a float where that is convenient
(comparison, arithmetic in an interaction equation) but carries the full audit
trail, and :meth:`StrengthResult.report` prints it as a calculation sheet.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field

from .citations import Citation
from .config import Basis
from .enums import LimitState
from .exceptions import AISC360Error

__all__ = ["LimitStateResult", "StrengthResult", "InteractionResult"]


@dataclass(frozen=True, slots=True)
class LimitStateResult:
    """Nominal strength from one limit state.

    Attributes
    ----------
    limit_state:
        Which failure mode this is.
    nominal:
        Nominal strength Rn in the parent result's units. Never factored.
    citation:
        The equation this came from.
    detail:
        Intermediate values worth showing on a calc sheet (Fe, Lp, Lr, lambda,
        Cb, ...). Purely informational; nothing reads it back.
    note:
        Free text, e.g. "Lb <= Lp, LTB does not apply".
    phi, omega:
        Factors specific to *this* limit state, overriding the parent result's.
        Most chapters state one pair for the whole chapter and leave these
        ``None``; Chapters D and J do not. Sect. D2, p. 16.1-28, gives tensile
        yielding phi = 0.90 / Omega = 1.67 but tensile rupture phi = 0.75 /
        Omega = 2.00 -- so the two limit states cannot be compared on nominal
        strength alone. See :attr:`StrengthResult.governing`.
    """

    limit_state: LimitState
    nominal: float
    citation: Citation
    detail: Mapping[str, float] = field(default_factory=dict)
    note: str = ""
    phi: float | None = None
    omega: float | None = None

    def __post_init__(self) -> None:
        if self.nominal < 0.0:
            raise AISC360Error(
                f"negative nominal strength {self.nominal} for {self.limit_state}"
            )
        if (self.phi is None) != (self.omega is None):
            raise AISC360Error(
                f"{self.limit_state}: supply both phi and omega, or neither -- "
                "a result must be readable under either design philosophy"
            )


@dataclass(frozen=True)
class StrengthResult:
    """Available strength, the governing limit state, and the full workings.

    Attributes
    ----------
    symbol:
        ``"Pn"``, ``"Mn"``, ``"Vn"``, ``"Rn"`` -- as the Specification names it.
    unit:
        Display unit, e.g. ``"kip"`` or ``"kip-in."``.
    phi, omega:
        Resistance and safety factors for this chapter. Both are always carried
        so a result computed under LRFD can be re-read under ASD.
    basis:
        Which of the two was applied to produce :attr:`available`.
    limit_states:
        Every limit state evaluated, in the order the chapter lists them.
    """

    symbol: str
    unit: str
    phi: float
    omega: float
    basis: Basis
    limit_states: tuple[LimitStateResult, ...]

    def __post_init__(self) -> None:
        if not self.limit_states:
            raise AISC360Error(f"{self.symbol}: no limit states were evaluated")

    # -- core numbers -----------------------------------------------------
    def factors_for(self, state: LimitStateResult) -> tuple[float, float]:
        """``(phi, omega)`` applying to one limit state -- its own, or the chapter's."""
        if state.phi is not None and state.omega is not None:
            return state.phi, state.omega
        return self.phi, self.omega

    def available_of(self, state: LimitStateResult) -> float:
        """Available strength of one limit state under the active basis."""
        phi, omega = self.factors_for(state)
        return phi * state.nominal if self.basis.is_lrfd else state.nominal / omega

    @property
    def governing(self) -> LimitStateResult:
        """The limit state producing the lowest **available** strength.

        Comparing on available rather than nominal strength matters wherever a
        chapter assigns different factors to different limit states. Sect. D2
        does exactly that -- tensile yielding at phi = 0.90, tensile rupture at
        phi = 0.75 -- so the two orderings genuinely disagree when
        ``0.833 < Fy*Ag/(Fu*Ae) < 1``: rupture has the larger nominal strength
        but the smaller design strength, and it is the design strength that must
        satisfy Eq. B3-1.

        Where a chapter uses one pair throughout (E, F, G), the two orderings
        are identical and this reduces to picking the lowest ``Rn``.

        Ties resolve to the first listed, which is the order the Specification
        presents them in -- so yielding is reported over buckling at Lb = Lp,
        matching how the Design Examples describe that boundary case.
        """
        return min(self.limit_states, key=self.available_of)

    @property
    def nominal(self) -> float:
        """Nominal strength of the governing limit state, Rn."""
        return self.governing.nominal

    @property
    def available(self) -> float:
        """phi*Rn under LRFD, or Rn/Omega under ASD (Eq. B3-1 / B3-2)."""
        return self.available_of(self.governing)

    @property
    def citation(self) -> Citation:
        """Citation of the governing limit state."""
        return self.governing.citation

    @property
    def limit_state(self) -> LimitState:
        """Which limit state governed."""
        return self.governing.limit_state

    def as_basis(self, basis: Basis) -> StrengthResult:
        """The same nominal strengths, evaluated under the other philosophy."""
        return StrengthResult(
            symbol=self.symbol,
            unit=self.unit,
            phi=self.phi,
            omega=self.omega,
            basis=basis,
            limit_states=self.limit_states,
        )

    def utilization(self, required: float) -> float:
        """Demand-capacity ratio: required / available.

        ``required`` is Ru under LRFD or Ra under ASD -- the caller is
        responsible for having factored the loads to match :attr:`basis`.
        """
        if self.available == 0.0:
            raise AISC360Error(f"{self.symbol}: available strength is zero")
        return required / self.available

    def is_adequate(self, required: float) -> bool:
        """True when the member satisfies Eq. B3-1 (LRFD) or B3-2 (ASD)."""
        return self.utilization(required) <= 1.0

    # -- float-like behaviour ---------------------------------------------
    def __float__(self) -> float:
        return self.available

    def __lt__(self, other: float) -> bool:
        return self.available < float(other)

    def __le__(self, other: float) -> bool:
        return self.available <= float(other)

    # -- reporting ---------------------------------------------------------
    def report(self, required: float | None = None, width: int = 78) -> str:
        """Render the full limit-state table as a plain-text calculation sheet."""
        factor = (
            f"phi = {self.phi:.2f}" if self.basis.is_lrfd else f"Omega = {self.omega:.2f}"
        )
        avail_symbol = (
            f"phi*{self.symbol}" if self.basis.is_lrfd else f"{self.symbol}/Omega"
        )

        lines: list[str] = []
        lines.append("=" * width)
        lines.append(f"{self.symbol} -- {self.basis.value} ({factor})")
        lines.append("=" * width)
        lines.append(f"{'limit state':<44}{self.symbol + ' [' + self.unit + ']':>20}{'':>14}")
        lines.append("-" * width)

        governing = self.governing
        # When a chapter varies its factors by limit state (Sects. D2, J), the
        # nominal column alone does not explain which state governs -- so the
        # available column is shown too.
        mixed = any(ls.phi is not None for ls in self.limit_states)

        for ls in self.limit_states:
            mark = "  <-- governs" if ls is governing else ""
            if mixed:
                phi, omega = self.factors_for(ls)
                shown = f"{phi:.2f}" if self.basis.is_lrfd else f"{omega:.2f}"
                lines.append(
                    f"{ls.limit_state.value:<34}{ls.nominal:>13,.4g}"
                    f"{shown:>7}{self.available_of(ls):>13,.4g}{mark:>11}"
                )
            else:
                lines.append(f"{ls.limit_state.value:<44}{ls.nominal:>20,.4g}{mark:>14}")
            if ls.note:
                lines.append(f"    {ls.note}")

        lines.append("-" * width)
        lines.append(f"{'governing: ' + governing.citation.text:<44}")
        lines.append(f"{self.symbol:<44}{self.nominal:>20,.4g}")
        lines.append(f"{avail_symbol:<44}{self.available:>20,.4g}")

        if required is not None:
            ratio = self.utilization(required)
            verdict = "OK" if ratio <= 1.0 else "NOT OK"
            lines.append("-" * width)
            lines.append(f"{self.basis.demand_symbol:<44}{required:>20,.4g}")
            lines.append(f"{'demand / capacity':<44}{ratio:>20.3f}   {verdict}")

        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"<{self.symbol}={self.nominal:.4g} {self.unit} "
            f"({self.basis.value} {self.available:.4g}) "
            f"governed by {self.limit_state.value}>"
        )

    # -- construction ------------------------------------------------------
    @classmethod
    def build(
        cls,
        symbol: str,
        unit: str,
        limit_states: Iterable[LimitStateResult | None],
        *,
        phi: float,
        omega: float,
        basis: Basis,
    ) -> StrengthResult:
        """Assemble a result, dropping limit states that do not apply.

        A chapter evaluates every branch and passes ``None`` for the ones its
        applicability rules exclude, keeping the call site declarative.
        """
        states: Sequence[LimitStateResult] = tuple(ls for ls in limit_states if ls is not None)
        return cls(
            symbol=symbol,
            unit=unit,
            phi=phi,
            omega=omega,
            basis=basis,
            limit_states=tuple(states),
        )


@dataclass(frozen=True)
class InteractionResult:
    """An interaction check: a demand-capacity ratio that must not exceed 1.0.

    Chapters H and J state several provisions as *inequalities* rather than as
    strengths -- Eqs. H1-1a, H1-1b, H1-3, H2-1, H3-6 and H4-1 all read
    ``something <= 1.0``. There is no ``Rn`` to report and no phi to apply,
    because the factors are already inside the ``Pc``, ``Mc`` and ``Vc`` terms
    the caller supplies.

    Attributes
    ----------
    ratio:
        Left-hand side of the interaction equation. Adequate at or below 1.0.
    citation:
        The equation applied -- which branch was taken is itself information.
    terms:
        Each contribution, for the calculation sheet, e.g. ``{"Pr/Pc": 0.42}``.
    note:
        Why this branch, or what the caller must still verify.
    """

    ratio: float
    citation: Citation
    limit_state: LimitState = LimitState.COMBINED_AXIAL_FLEXURE
    terms: Mapping[str, float] = field(default_factory=dict)
    note: str = ""

    def __post_init__(self) -> None:
        if self.ratio < 0.0:
            raise AISC360Error(f"interaction ratio must be non-negative, got {self.ratio}")

    @property
    def is_adequate(self) -> bool:
        """True when the interaction equation is satisfied."""
        return self.ratio <= 1.0

    @property
    def margin(self) -> float:
        """Fraction of capacity remaining; negative when overstressed."""
        return 1.0 - self.ratio

    def __float__(self) -> float:
        return self.ratio

    def report(self, width: int = 78) -> str:
        """Render the interaction check as a plain-text calculation sheet."""
        lines = ["=" * width, f"{self.limit_state.value} -- interaction check", "=" * width]
        for name, value in self.terms.items():
            lines.append(f"{name:<58}{value:>20,.4f}")
        lines.append("-" * width)
        lines.append(f"{self.citation.text:<58}")
        verdict = "OK" if self.is_adequate else "NOT OK"
        lines.append(f"{'interaction ratio':<58}{self.ratio:>16,.4f}   {verdict}")
        if self.note:
            lines.append(f"    {self.note}")
        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        verdict = "OK" if self.is_adequate else "NOT OK"
        return f"<{self.citation.equation}: {self.ratio:.4f} {verdict}>"
