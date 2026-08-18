"""Chapter C -- Design for Stability.

Chapter C computes no strengths, so the tests focus on the three things it does
produce and on the couplings out of it:

* the ``tau_b`` curve, swept across its full domain and pinned at every feature;
* the notional loads, including the Sect. C2.3(c) alternative;
* the Appendix 8 coupling -- reduced stiffness must inflate ``B1``.

Precision: machine precision throughout. Every expression here is exact -- no
rounded coefficients, unlike Eqs. E3-2's 4.71 or F2-6's 1.95/6.76.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from pyaisc360 import appendix_7 as A7
from pyaisc360 import appendix_8 as A8
from pyaisc360 import chapter_c as C
from pyaisc360.core.config import DEFAULT_SETTINGS, Basis, DesignSettings
from pyaisc360.core.enums import StabilityMethod
from pyaisc360.core.exceptions import AISC360Error, GeometryError, OutOfScopeError

E_STEEL = 29000.0
PNS = 1000.0


# ===========================================================================
# Sect. C1 -- general stability requirements
# ===========================================================================
class TestC1Gates:
    def test_p_delta_relief_needs_all_three_conditions(self) -> None:
        """Sect. C2.1(b), p. 16.1-23. Source: AISC Sect. C2.1."""
        ok, _ = C.p_delta_may_be_neglected(1.5, 0.25)
        assert ok

    def test_drift_above_1_7_blocks_the_relief(self) -> None:
        ok, reason = C.p_delta_may_be_neglected(1.8, 0.25)
        assert not ok
        assert "C2.1(b)(2)" in reason

    def test_too_much_gravity_on_moment_frames_blocks_it(self) -> None:
        """Sect. C2.1(b)(3): no more than one-third."""
        assert C.p_delta_may_be_neglected(1.5, 1.0 / 3.0)[0]
        ok, reason = C.p_delta_may_be_neglected(1.5, 0.40)
        assert not ok
        assert "one-third" in reason

    def test_non_vertical_gravity_system_blocks_it(self) -> None:
        ok, reason = C.p_delta_may_be_neglected(
            1.5, 0.25, gravity_through_vertical_elements=False
        )
        assert not ok
        assert "C2.1(b)(1)" in reason

    def test_the_member_caveat_is_never_dropped(self) -> None:
        """Sect. C2.1(b) closes: "It is necessary in ALL CASES to consider
        P-delta effects in the evaluation of individual members subject to
        compression and flexure." The structure-level relief is not a member
        relief -- B1 is still required."""
        _, note = C.p_delta_may_be_neglected(1.0, 0.1)
        assert "B1 multiplier" in note

    def test_the_three_drift_thresholds_are_distinct(self) -> None:
        """Chapter C uses 1.7; Appendix 7 uses 1.5 to gate its methods and 1.1
        to skip a sidesway buckling analysis. Three numbers, three questions."""
        assert C.P_DELTA_DRIFT_LIMIT == 1.7
        assert A7.DRIFT_RATIO_LIMIT == 1.5
        assert A7.K_EQUAL_ONE_DRIFT_LIMIT == 1.1
        assert C.P_DELTA_DRIFT_LIMIT > A7.DRIFT_RATIO_LIMIT > A7.K_EQUAL_ONE_DRIFT_LIMIT


# ===========================================================================
# Sect. C2.2 -- notional loads
# ===========================================================================
class TestNotionalLoads:
    def test_eq_C2_1_lrfd(self) -> None:
        """Ni = 0.002*alpha*Yi. Source: AISC Sect. C2.2b."""
        assert C.notional_load(1000.0, Basis.LRFD) == pytest.approx(2.0, rel=1e-12)

    def test_alpha_is_inside_the_equation(self) -> None:
        """Eq. C2-1 carries alpha; it is not applied afterwards. ASD gives 1.6x."""
        lrfd = C.notional_load(1000.0, Basis.LRFD)
        asd = C.notional_load(1000.0, Basis.ASD)
        assert asd == pytest.approx(1.6 * lrfd, rel=1e-12)
        assert asd == pytest.approx(3.2, rel=1e-12)

    def test_the_coefficient_is_the_out_of_plumbness(self) -> None:
        """Sect. C2.2b(c): 0.002 comes from 1/500, and may be adjusted
        proportionally where a different tolerance is justified."""
        assert pytest.approx(
            C.OUT_OF_PLUMBNESS, rel=1e-12
        ) == C.NOTIONAL_LOAD_COEFFICIENT
        tighter = C.notional_load(1000.0, out_of_plumbness=1.0 / 1000.0)
        assert tighter == pytest.approx(0.5 * C.notional_load(1000.0), rel=1e-12)

    def test_gravity_only_relief_at_1_7(self) -> None:
        """Sect. C2.2b(d). Source: AISC Sect. C2.2b."""
        allowed, note = C.notional_loads_gravity_only(1.7)
        assert allowed
        assert "gravity-only" in note
        blocked, note = C.notional_loads_gravity_only(1.71)
        assert not blocked
        assert "ALL load combinations" in note

    def test_the_stiffness_basis_differs_from_appendix_7(self) -> None:
        """Sect. C2.2b(d) judges the drift ratio WITH the Sect. C2.3 reductions
        applied; Sects. 7.2.1(b)/7.3.1(b) judge theirs WITHOUT. Same-looking
        test, different basis -- recorded in the docstring so it is not lost."""
        assert "adjusted" in C.notional_loads_gravity_only.__doc__

    def test_negative_gravity_load_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="non-negative"):
            C.notional_load(-100.0)


class TestTauBUnityAlternative:
    def test_the_additional_notional_load(self) -> None:
        """Sect. C2.3(c): 0.001*alpha*Yi. Source: AISC Sect. C2.3."""
        assert C.notional_load_for_tau_b_unity(1000.0) == pytest.approx(1.0, rel=1e-12)

    def test_it_is_half_the_Eq_C2_1_load(self) -> None:
        """0.001 against 0.002 -- and it is ADDITIVE, so a frame using this
        option carries 0.003*alpha*Yi in total, not 0.001."""
        assert C.notional_load_for_tau_b_unity(1000.0) == pytest.approx(
            0.5 * C.notional_load(1000.0), rel=1e-12
        )
        total = C.notional_load(1000.0) + C.notional_load_for_tau_b_unity(1000.0)
        assert total == pytest.approx(3.0, rel=1e-12)

    def test_it_scales_with_alpha_too(self) -> None:
        assert C.notional_load_for_tau_b_unity(1000.0, Basis.ASD) == pytest.approx(
            1.6, rel=1e-12
        )

    def test_the_option_buys_tau_b_of_one(self) -> None:
        """With the option, a heavily loaded member keeps its full flexural
        stiffness -- which is the whole point: it removes the tau_b iteration."""
        without = C.reduced_stiffness(E_STEEL, 800.0, 26.5, 900.0, PNS)
        with_option = C.reduced_stiffness(
            E_STEEL, 800.0, 26.5, 900.0, PNS, tau_b_unity=True
        )
        assert without.tau_b == pytest.approx(0.36, rel=1e-12)
        assert with_option.tau_b == 1.0
        assert with_option.EI_star > without.EI_star
        assert "C2.3(c)" in with_option.note


# ===========================================================================
# Sect. C2.3 -- tau_b
# ===========================================================================
class TestTauBCurve:
    """The full sweep the brief asked for, plus every feature of the curve."""

    @pytest.mark.parametrize("ratio", [0.0, 0.1, 0.25, 0.4, 0.49, 0.5])
    def test_exactly_unity_up_to_and_including_the_threshold(self, ratio: float) -> None:
        """Eq. C2-2a holds for alpha*Pr/Pns <= 0.5, inclusive. Exactly 1.0 --
        not approximately. Source: AISC Sect. C2.3."""
        assert C.tau_b(ratio * PNS, PNS) == 1.0

    def test_the_branches_meet_at_the_threshold_to_machine_precision(self) -> None:
        """4*(0.5)*(1 - 0.5) = 1.0 exactly, so Eq. C2-2b's value at the
        threshold equals Eq. C2-2a's."""
        at = C.tau_b(0.5 * PNS, PNS)
        just_above = C.tau_b(0.5 * PNS * (1 + 1e-15), PNS)
        assert at == 1.0
        assert abs(just_above - 1.0) < 1e-14

    def test_the_derivative_is_also_continuous_at_the_threshold(self) -> None:
        """d/dx[4x(1-x)] = 4 - 8x, which is zero at x = 0.5.

        The parabola's PEAK sits exactly on the transition, so tau_b is C1
        continuous -- no kink. That is why successive substitution on tau_b
        converges without damping: there is no discontinuity in slope for an
        iteration to oscillate across.
        """
        eps = 1e-6
        below_slope = 0.0  # Eq. C2-2a is constant
        above = C.tau_b((0.5 + eps) * PNS, PNS)
        at = C.tau_b(0.5 * PNS, PNS)
        above_slope = (above - at) / eps
        assert abs(above_slope - below_slope) < 1e-4

    @pytest.mark.parametrize(
        ("ratio", "expected"),
        [(0.6, 0.96), (0.7, 0.84), (0.75, 0.75), (0.8, 0.64), (0.9, 0.36), (0.95, 0.19)],
    )
    def test_the_parabola_above_the_threshold(self, ratio: float, expected: float) -> None:
        """Eq. C2-2b: tau_b = 4x(1-x). Source: AISC Sect. C2.3."""
        assert C.tau_b(ratio * PNS, PNS) == pytest.approx(expected, abs=1e-12)

    def test_it_reaches_exactly_zero_at_the_squash_load(self) -> None:
        """4*(1.0)*(1 - 1.0) = 0 exactly -- to machine precision, not near it."""
        assert C.tau_b(PNS, PNS) == pytest.approx(0.0, abs=2e-16)

    def test_it_decreases_monotonically_above_the_threshold(self) -> None:
        values = [C.tau_b(r / 100.0 * PNS, PNS) for r in range(50, 101)]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))

    def test_it_never_exceeds_unity_anywhere(self) -> None:
        for r in range(0, 101):
            assert C.tau_b(r / 100.0 * PNS, PNS) <= 1.0

    def test_it_never_goes_negative(self) -> None:
        """The parabola would go negative above 1.0; the floor and the raise
        together make that unreachable."""
        for r in range(0, 101):
            assert C.tau_b(r / 100.0 * PNS, PNS) >= 0.0

    def test_above_the_squash_load_raises(self) -> None:
        """alpha*Pr > Pns means the member has exceeded its cross-section
        compressive strength. Eq. C2-2b would return a negative stiffness
        multiplier -- meaningless, so it raises rather than flooring silently."""
        with pytest.raises(OutOfScopeError, match="exceeds 1.0"):
            C.tau_b(1.01 * PNS, PNS)

    def test_asd_alpha_moves_the_cliff(self) -> None:
        """alpha = 1.6 means the threshold is reached at Pr/Pns = 0.3125, not
        0.5 -- so ASD bites where LRFD does not."""
        Pr = 0.4 * PNS
        assert C.tau_b(Pr, PNS, Basis.LRFD) == 1.0
        assert C.tau_b(Pr, PNS, Basis.ASD) == pytest.approx(4 * 0.64 * 0.36, rel=1e-12)
        assert C.tau_b(0.3125 * PNS, PNS, Basis.ASD) == 1.0

    def test_tension_members_keep_full_stiffness(self) -> None:
        """Sect. C2.3(b) models partial yielding under COMPRESSION."""
        assert C.tau_b(-500.0, PNS) == 1.0

    def test_non_positive_Pns_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="Pns must be positive"):
            C.tau_b(500.0, 0.0)

    def test_Pns_is_the_cross_section_strength_not_the_column_strength(self) -> None:
        """Sect. C2.3(b) and Eq. A-7-1 use the same Pns: Fy*Ag for a nonslender
        section. No buckling reduction -- tau_b models partial yielding of the
        SECTION, not member buckling."""
        assert "cross-section" in C.tau_b.__doc__
        assert A7.axial_force_limit(PNS) == pytest.approx(0.5 * PNS, rel=1e-12)


class TestStiffnessReduction:
    def test_the_0_8_factor(self) -> None:
        """Sect. C2.3(a). Source: AISC Sect. C2.3."""
        assert C.STIFFNESS_REDUCTION == 0.80

    def test_EI_star_is_0_8_tau_b_EI(self) -> None:
        """The Sect. C2.3 User Note states it directly."""
        EI = E_STEEL * 800.0
        got = C.reduced_flexural_stiffness(E_STEEL, 800.0, 900.0, PNS)
        assert got == pytest.approx(0.8 * 0.36 * EI, rel=1e-12)

    def test_EA_star_has_no_tau_b(self) -> None:
        """Sect. C2.3(b) applies tau_b to FLEXURAL stiffness only. Applying it to
        axial stiffness as well over-softens the frame."""
        assert C.reduced_axial_stiffness(E_STEEL, 26.5) == pytest.approx(
            0.8 * E_STEEL * 26.5, rel=1e-12
        )

    def test_axial_reduction_is_independent_of_the_axial_load(self) -> None:
        light = C.reduced_stiffness(E_STEEL, 800.0, 26.5, 100.0, PNS)
        heavy = C.reduced_stiffness(E_STEEL, 800.0, 26.5, 900.0, PNS)
        assert light.EA_star == pytest.approx(heavy.EA_star, rel=1e-12)
        assert heavy.EI_star < light.EI_star

    def test_the_combined_flexural_factor(self) -> None:
        rs = C.reduced_stiffness(E_STEEL, 800.0, 26.5, 900.0, PNS)
        assert rs.flexural_reduction == pytest.approx(0.8 * 0.36, rel=1e-12)

    def test_below_the_threshold_the_only_reduction_is_0_8(self) -> None:
        rs = C.reduced_stiffness(E_STEEL, 800.0, 26.5, 300.0, PNS)
        assert rs.tau_b == 1.0
        assert rs.flexural_reduction == pytest.approx(0.8, rel=1e-12)
        assert "C2-2a" in rs.note

    def test_report_renders(self) -> None:
        text = C.reduced_stiffness(E_STEEL, 800.0, 26.5, 900.0, PNS).report()
        assert "tau_b" in text and "EI*" in text and "EA*" in text


# ===========================================================================
# Sect. C3 -- available strengths
# ===========================================================================
class TestC3EffectiveLength:
    def test_direct_analysis_forces_K_to_unity(self) -> None:
        """Sect. C3, p. 16.1-27. Source: AISC Sect. C3."""
        assert C.effective_length_factor(StabilityMethod.DIRECT_ANALYSIS) == 1.0

    def test_a_smaller_K_is_permitted(self) -> None:
        """Sect. C3: "unless a SMALLER value is justified by rational analysis"."""
        assert C.effective_length_factor(StabilityMethod.DIRECT_ANALYSIS, K=0.8) == 0.8

    def test_a_larger_K_is_rejected(self) -> None:
        """K > 1 under the direct analysis method double-counts: the stability it
        used to represent is already in the reduced stiffness and notional loads."""
        with pytest.raises(OutOfScopeError, match="double-counts"):
            C.effective_length_factor(StabilityMethod.DIRECT_ANALYSIS, K=2.0)

    def test_first_order_method_also_caps_at_unity(self) -> None:
        """Sect. 7.3.3 takes Lc as the unbraced length, same as Sect. C3."""
        assert C.effective_length_factor(StabilityMethod.FIRST_ORDER) == 1.0
        with pytest.raises(OutOfScopeError):
            C.effective_length_factor(StabilityMethod.FIRST_ORDER, K=1.5)

    def test_effective_length_method_permits_K_above_unity(self) -> None:
        """The opposite rule: Sect. 7.2.3(b) gets K from a sidesway buckling
        analysis, and K > 1 is the whole point of that method."""
        assert C.effective_length_factor(StabilityMethod.EFFECTIVE_LENGTH, K=2.1) == 2.1

    def test_effective_length_method_requires_an_explicit_K(self) -> None:
        with pytest.raises(AISC360Error, match="sidesway buckling"):
            C.effective_length_factor(StabilityMethod.EFFECTIVE_LENGTH)

    def test_non_positive_K_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="K must be positive"):
            C.effective_length_factor(StabilityMethod.DIRECT_ANALYSIS, K=0.0)


class TestASDAnalysisScale:
    def test_lrfd_needs_no_scaling(self) -> None:
        assert C.asd_analysis_scale(Basis.LRFD) == 1.0

    def test_asd_runs_at_1_6_and_divides_back(self) -> None:
        """Sect. C2.1(d), p. 16.1-24. The two operations do NOT cancel, because
        second-order analysis is non-linear -- which is exactly why the
        Specification spells out both halves."""
        assert C.asd_analysis_scale(Basis.ASD) == 1.6
        assert "non-linear" in C.asd_analysis_scale.__doc__


# ===========================================================================
# Coupling into Appendix 8
# ===========================================================================
class TestAppendix8Coupling:
    """The pipeline the brief asked for: Chapter C stiffness -> Eq. A-8-5 -> B1."""

    E, I, LC1, CM = E_STEEL, 800.0, 180.0, 1.0

    def _B1(self, Pr: float, *, reduced: bool) -> float:
        if reduced:
            EI_star = C.reduced_flexural_stiffness(self.E, self.I, Pr, PNS)
        else:
            EI_star = self.E * self.I
        return A8.B1_multiplier(self.CM, Pr, A8.Pe1(EI_star, self.LC1), Basis.LRFD)

    def test_the_0_8_factor_alone_inflates_B1(self) -> None:
        """Even below the tau_b threshold, the 0.8 lowers Pe1 and raises B1."""
        Pr = 300.0  # alpha*Pr/Pns = 0.30, tau_b = 1.0
        assert C.tau_b(Pr, PNS) == 1.0
        assert self._B1(Pr, reduced=True) > self._B1(Pr, reduced=False)

    def test_tau_b_inflates_B1_much_further(self) -> None:
        """At alpha*Pr/Pns = 0.8, tau_b = 0.64 and the combined factor is 0.512
        -- so Pe1 halves and B1 climbs sharply."""
        Pr = 800.0
        assert C.tau_b(Pr, PNS) == pytest.approx(0.64, rel=1e-12)
        nominal = self._B1(Pr, reduced=False)
        reduced = self._B1(Pr, reduced=True)
        assert reduced > nominal
        assert reduced / nominal > 1.1

    def test_the_inflation_grows_with_the_axial_ratio(self) -> None:
        ratios = [
            self._B1(r * PNS, reduced=True) / self._B1(r * PNS, reduced=False)
            for r in (0.3, 0.6, 0.75, 0.85)
        ]
        assert all(a <= b for a, b in zip(ratios, ratios[1:], strict=False))

    def test_Pe1_scales_exactly_with_the_stiffness_factor(self) -> None:
        """Eq. A-8-5 is linear in EI*, so Pe1 tracks 0.8*tau_b exactly."""
        Pr = 900.0
        factor = C.reduced_stiffness(self.E, self.I, 26.5, Pr, PNS).flexural_reduction
        nominal = A8.Pe1(self.E * self.I, self.LC1)
        reduced = A8.Pe1(
            C.reduced_flexural_stiffness(self.E, self.I, Pr, PNS), self.LC1
        )
        assert reduced / nominal == pytest.approx(factor, rel=1e-12)

    def test_the_tau_b_unity_option_recovers_the_lost_amplification(self) -> None:
        """Sect. C2.3(c) trades the tau_b reduction for a notional load, so B1
        falls back to the 0.8-only value."""
        Pr = 800.0
        with_tau_b = self._B1(Pr, reduced=True)
        EI_unity = C.reduced_flexural_stiffness(
            self.E, self.I, Pr, PNS, tau_b_unity=True
        )
        without = A8.B1_multiplier(self.CM, Pr, A8.Pe1(EI_unity, self.LC1), Basis.LRFD)
        assert without < with_tau_b
        assert EI_unity == pytest.approx(0.8 * self.E * self.I, rel=1e-12)

    def test_a_heavily_loaded_short_column_can_reach_the_singularity(self) -> None:
        """Softening the member enough drives alpha*Pr towards Pe1, where
        Eq. A-8-3 is singular -- Appendix 8 raises rather than returning a huge
        number, and the two chapters agree on that being a design failure."""
        with pytest.raises(OutOfScopeError, match="elastic buckling load"):
            EI_star = C.reduced_flexural_stiffness(E_STEEL, 20.0, 950.0, PNS)
            A8.B1_multiplier(1.0, 950.0, A8.Pe1(EI_star, 400.0), Basis.LRFD)


# ===========================================================================
# DesignSettings coupling
# ===========================================================================
class TestDesignSettingsCoupling:
    def test_direct_analysis_is_the_default(self) -> None:
        """It is the only method with no limitations, so it is the safe default."""
        assert DEFAULT_SETTINGS.stability_method is StabilityMethod.DIRECT_ANALYSIS

    def test_the_flag_survives_a_basis_switch(self) -> None:
        settings = DesignSettings(stability_method=StabilityMethod.EFFECTIVE_LENGTH)
        assert settings.with_basis(Basis.ASD).stability_method is (
            StabilityMethod.EFFECTIVE_LENGTH
        )

    def test_appendix_7_shares_the_same_enum(self) -> None:
        """One enum, not two -- so a settings object and an Appendix 7 check
        cannot disagree about which method is in use."""
        assert A7.StabilityMethod is StabilityMethod

    def test_settings_drive_the_effective_length_rule(self) -> None:
        for method, K, expected in (
            (StabilityMethod.DIRECT_ANALYSIS, None, 1.0),
            (StabilityMethod.FIRST_ORDER, None, 1.0),
            (StabilityMethod.EFFECTIVE_LENGTH, 1.7, 1.7),
        ):
            settings = DesignSettings(stability_method=method)
            assert C.effective_length_factor(settings.stability_method, K=K) == expected


# ===========================================================================
# Provenance
# ===========================================================================
_EXAMPLES = Path(__file__).parent / "design_examples" / "chapter_c.json"


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
    return [case for case in data.get("cases", []) if not case.get("skip")]


class TestProvenance:
    @pytest.mark.design_example
    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: str(c.get("id", "?")))
    def test_case(self, case: dict[str, Any]) -> None:
        basis = Basis(str(case.get("basis", "LRFD")))
        tol = float(case.get("tol", 1e-12))
        kind = case["kind"]

        if kind == "tau_b":
            got = C.tau_b(float(case["Pr"]), float(case["Pns"]), basis)
        elif kind == "notional":
            got = C.notional_load(float(case["Yi"]), basis)
        elif kind == "notional_tau_b_unity":
            got = C.notional_load_for_tau_b_unity(float(case["Yi"]), basis)
        else:  # pragma: no cover
            raise AssertionError(f"unknown case kind {kind!r}")

        expected = float(case["expected"])
        if tol == 0.0:
            assert got == expected
        else:
            assert got == pytest.approx(expected, abs=tol)

    def test_every_case_declares_its_source(self) -> None:
        """The provenance rule: no expected value without a stated source."""
        for case in _load_cases():
            assert case.get("source"), f"case {case.get('id')!r} has no source"

    def test_sources_name_the_governing_section(self) -> None:
        valid = {"AISC Sect. C2.1", "AISC Sect. C2.2b", "AISC Sect. C2.3", "AISC Sect. C3"}
        for case in _load_cases():
            assert case["source"] in valid, case["id"]
