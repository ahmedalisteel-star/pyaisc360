"""Chapter I -- Composite Members.

Composite design mixes two materials with different factors, different failure
modes and different sign conventions, so the tests concentrate on the places
where a steel-only habit gives the wrong answer: the compression factor, the
squash-load form of the buckling equations, the anchor reduction factors, and
the two interpolation rules that are deliberately different from each other.

Precision: 1e-12 for closed forms, 0.5% where a value is traceable to a rounded
published figure.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pyaisc360 import chapter_e as E
from pyaisc360 import chapter_i as I
from pyaisc360 import chapter_j as J
from pyaisc360.core.enums import FlexuralSlenderness
from pyaisc360.core.exceptions import AISC360Error, OutOfScopeError

ES = 29000.0
COMPACT = FlexuralSlenderness.COMPACT
NONCOMPACT = FlexuralSlenderness.NONCOMPACT
SLENDER = FlexuralSlenderness.SLENDER


# ===========================================================================
# Sect. I1
# ===========================================================================
class TestMaterialLimits:
    def test_normal_weight_concrete_band(self) -> None:
        """Sect. I1.3(a): 3 <= f'c <= 10 ksi for normal weight."""
        assert I.check_material_limits(4.0, 50.0) == []
        assert I.check_material_limits(10.0, 50.0) == []
        assert len(I.check_material_limits(12.0, 50.0)) == 1

    def test_lightweight_ceiling_is_lower(self) -> None:
        """6 ksi for lightweight against 10 for normal weight."""
        assert I.check_material_limits(8.0, 50.0, lightweight=False) == []
        assert len(I.check_material_limits(8.0, 50.0, lightweight=True)) == 1

    def test_the_fc_cap_applies_only_to_strength_calculations(self) -> None:
        """Sect. I1.3(a) User Note: higher f'c "may be used for stiffness
        calculations but may not be relied upon for strength". Asymmetric."""
        assert len(I.check_material_limits(12.0, 50.0, for_strength=True)) == 1
        assert I.check_material_limits(12.0, 50.0, for_strength=False) == []

    def test_the_3_ksi_floor_applies_either_way(self) -> None:
        """A weaker concrete is not covered for stiffness either."""
        assert len(I.check_material_limits(2.0, 50.0, for_strength=False)) == 1

    def test_yield_stress_caps(self) -> None:
        """Sect. I1.3(b): Fy <= 75 ksi; I1.3(c): Fysr <= 80 ksi."""
        assert I.check_material_limits(4.0, 75.0, 80.0) == []
        assert len(I.check_material_limits(4.0, 80.0, 80.0)) == 1
        assert len(I.check_material_limits(4.0, 50.0, 90.0)) == 1

    def test_violations_report_rather_than_raise(self) -> None:
        """A violation makes the equations inapplicable, not the member illegal."""
        violations = I.check_material_limits(2.0, 90.0, 90.0)
        assert len(violations) == 3


class TestConcreteModulus:
    def test_normal_weight_4ksi(self) -> None:
        """Ec = 145^1.5*sqrt(4) = 3492 ksi -- about E/8.3."""
        assert I.concrete_modulus(145.0, 4.0) == pytest.approx(3492.0, rel=0.005)

    def test_scales_with_the_square_root_of_fc(self) -> None:
        assert I.concrete_modulus(145.0, 8.0) / I.concrete_modulus(145.0, 2.0) == (
            pytest.approx(2.0, rel=1e-12)
        )

    def test_unit_weight_outside_the_range_is_rejected(self) -> None:
        with pytest.raises(OutOfScopeError, match="90-155"):
            I.concrete_modulus(200.0, 4.0)


class TestReinforcementRatio:
    def test_minimum_is_0_004_for_encased(self) -> None:
        """Eq. I2-1, Sect. I2.1a(c)."""
        rho, ok = I.reinforcement_ratio(1.6, 400.0)
        assert rho == pytest.approx(0.004, rel=1e-12)
        assert ok

    def test_just_below_the_minimum_fails(self) -> None:
        _, ok = I.reinforcement_ratio(1.5, 400.0)
        assert not ok

    def test_filled_members_need_no_minimum(self) -> None:
        """Sect. I2.2a(c): "minimum longitudinal reinforcement is not required"
        for filled members -- the tube already confines the concrete."""
        rho, ok = I.reinforcement_ratio(0.0, 400.0)
        assert rho == 0.0
        assert not ok  # the caller decides whether the minimum applies


# ===========================================================================
# Sect. I2
# ===========================================================================
class TestCompositeCompression:
    def test_the_factor_is_not_chapter_E_s(self) -> None:
        """Sect. I2 uses phi_c = 0.75 / Omega_c = 2.00, not 0.90 / 1.67.

        Using Chapter E's factor on a composite column overstates it by 20%.
        """
        assert (I.PHI_C_COMPOSITE, I.OMEGA_C_COMPOSITE) == (0.75, 2.00)
        assert I.PHI_C_COMPOSITE != E.PHI_C
        assert pytest.approx(1.5, rel=1e-12) == I.PHI_C_COMPOSITE * I.OMEGA_C_COMPOSITE
        assert pytest.approx(1.2, rel=1e-12) == E.PHI_C / I.PHI_C_COMPOSITE

    def test_Pno_encased(self) -> None:
        """Eq. I2-4: Pno = Fy*As + Fysr*Asr + 0.85*f'c*Ac."""
        got = I.Pno_encased(50.0, 26.5, 60.0, 4.0, 5.0, 500.0)
        assert got == pytest.approx(50 * 26.5 + 60 * 4.0 + 0.85 * 5.0 * 500.0, rel=1e-12)

    def test_the_inelastic_branch_mirrors_Eq_E3_2(self) -> None:
        """Eqs. I2-2/I2-3 use the same 0.658, 0.877 and 2.25 as Eqs. E3-2/E3-3,
        but switch on Pno/Pe (two loads) rather than Fy/Fe (two stresses)."""
        result = I.composite_compressive_strength(2000.0, 4000.0)
        assert result.nominal == pytest.approx(2000.0 * math.pow(0.658, 0.5), rel=1e-12)
        assert result.citation.equation == "I2-2"

    def test_the_elastic_branch(self) -> None:
        result = I.composite_compressive_strength(4000.0, 1000.0)
        assert result.nominal == pytest.approx(0.877 * 1000.0, rel=1e-12)
        assert result.citation.equation == "I2-3"

    def test_the_branches_meet_at_2_25(self) -> None:
        """0.658^2.25 = 0.3894 and 0.877/2.25 = 0.3898 -- a 0.1% step, the same
        rounding the Chapter E pair carries."""
        Pe = 1000.0
        Pno = 2.25 * Pe
        below = I.composite_compressive_strength(Pno, Pe).nominal
        above = I.composite_compressive_strength(Pno * 1.0001, Pe).nominal
        assert below == pytest.approx(above, rel=2e-3)

    def test_agrees_with_chapter_E_for_a_bare_steel_member(self) -> None:
        """With Pno = Fy*Ag and Pe = Fe*Ag the two ratios are identical, so
        Eq. I2-2 and Eq. E3-2 must return the same stress."""
        Ag, Fy = 26.5, 50.0
        Fe = E.elastic_buckling_stress(97.3, ES)
        by_chapter_e = E.flexural_buckling_stress(Fy, Fe) * Ag
        by_chapter_i = I.composite_compressive_strength(Fy * Ag, Fe * Ag).nominal
        assert by_chapter_i == pytest.approx(by_chapter_e, rel=1e-12)

    def test_the_bare_steel_floor(self) -> None:
        """Sects. I2.1b/I2.2b: "the available compressive strength need not be
        less than that specified for the bare steel member"."""
        result = I.composite_compressive_strength(2000.0, 4000.0, bare_steel_Pn=1800.0)
        assert result.nominal == pytest.approx(1800.0, rel=1e-12)
        assert "bare steel" in result.governing.note

    def test_C1_and_C3_differ_in_both_intercept_and_cap(self) -> None:
        """Eq. I2-7 (encased) is 0.25 + 3r <= 0.7; Eq. I2-13 (filled) is
        0.45 + 3r <= 0.9. A filled tube holds its concrete far better."""
        As, Asr, Ag = 20.0, 4.0, 400.0
        assert I.C1_coefficient(As, Asr, Ag) == pytest.approx(0.25 + 3 * 0.06, rel=1e-12)
        assert I.C3_coefficient(As, Asr, Ag) == pytest.approx(0.45 + 3 * 0.06, rel=1e-12)
        assert I.C1_coefficient(100.0, 20.0, 400.0) == 0.7
        assert I.C3_coefficient(100.0, 20.0, 400.0) == 0.9

    def test_reinforcement_takes_the_full_Es_Isr(self) -> None:
        """Eqs. I2-6/I2-12 discount only the CONCRETE by C1 or C3."""
        got = I.EIeff_encased(ES, 800.0, 100.0, 0.5, 3492.0, 5000.0)
        assert got == pytest.approx(ES * 800.0 + ES * 100.0 + 0.5 * 3492.0 * 5000.0, rel=1e-12)

    def test_Pe_composite(self) -> None:
        """Eq. I2-5: Pe = pi^2*EIeff/Lc^2."""
        assert I.Pe_composite(1.0e9, 180.0) == pytest.approx(
            math.pi**2 * 1.0e9 / 180.0**2, rel=1e-12
        )

    def test_tension_neglects_the_concrete(self) -> None:
        """Eqs. I2-8/I2-14: Pn = Fy*As + Fysr*Asr. Sect. I1.2 neglects concrete
        tension entirely."""
        result = I.composite_tensile_strength(50.0, 26.5, 60.0, 4.0)
        assert result.nominal == pytest.approx(50 * 26.5 + 60 * 4.0, rel=1e-12)
        assert (result.phi, result.omega) == (0.90, 1.67)


class TestFilledCompression:
    ARGS = dict(Fy=50.0, As=15.0, fc_prime=5.0, Ac=200.0, Asr=0.0, Es=ES, Ec=3900.0)

    def test_C2_is_higher_for_round_sections(self) -> None:
        """Eq. I2-9b: C2 = 0.85 rectangular, 0.95 round -- a round tube confines
        its concrete more effectively."""
        rect = I.Pp_filled(**self.ARGS, round_section=False)
        round_ = I.Pp_filled(**self.ARGS, round_section=True)
        assert round_ > rect
        assert (round_ - 50.0 * 15.0) / (rect - 50.0 * 15.0) == pytest.approx(
            0.95 / 0.85, rel=1e-12
        )

    def test_Py_uses_0_7_regardless_of_shape(self) -> None:
        """Eq. I2-9d uses 0.7, not C2 -- a noncompact tube buckles locally before
        the concrete reaches its stress block."""
        Py = I.Py_filled(50.0, 15.0, 5.0, 200.0, 0.0, ES, 3900.0)
        assert Py == pytest.approx(50 * 15.0 + 0.7 * 5.0 * 200.0, rel=1e-12)

    def test_the_reinforcement_is_transformed_not_added(self) -> None:
        """Filled members use Asr*Es/Ec inside the concrete term (Eq. I2-9b),
        where encased members add Fysr*Asr directly (Eq. I2-4). Genuinely
        different treatments of the same bars."""
        with_bars = I.Pp_filled(50.0, 15.0, 5.0, 200.0, 4.0, ES, 3900.0)
        without = I.Pp_filled(50.0, 15.0, 5.0, 200.0, 0.0, ES, 3900.0)
        assert with_bars - without == pytest.approx(
            0.85 * 5.0 * 4.0 * ES / 3900.0, rel=1e-12
        )

    def test_the_noncompact_interpolation_is_SQUARED(self) -> None:
        """Eq. I2-9c squares the slenderness fraction. Every other noncompact
        transition in the Specification -- F3-1, F4-13, F7-2, I3-3b -- is
        linear, so a linear reading here is unconservative across the band.
        """
        Pp, Py = 2000.0, 1600.0
        mid, note = I.Pno_filled(
            NONCOMPACT, Pp, Py, lam=15.0, lam_p=10.0, lam_r=20.0
        )
        assert mid == pytest.approx(Pp - (Pp - Py) * 0.5**2, rel=1e-12)
        assert "SQUARED" in note
        # a linear reading would have given 1800, not 1900
        assert mid == pytest.approx(1900.0, rel=1e-12)
        assert Pp - (Pp - Py) * 0.5 == pytest.approx(1800.0, rel=1e-12)

    def test_the_squared_interpolation_is_continuous_at_both_ends(self) -> None:
        Pp, Py = 2000.0, 1600.0
        at_p, _ = I.Pno_filled(NONCOMPACT, Pp, Py, lam=10.0, lam_p=10.0, lam_r=20.0)
        at_r, _ = I.Pno_filled(NONCOMPACT, Pp, Py, lam=20.0, lam_p=10.0, lam_r=20.0)
        assert at_p == pytest.approx(Pp, rel=1e-12)
        assert at_r == pytest.approx(Py, rel=1e-12)

    def test_compact_returns_Pp(self) -> None:
        got, note = I.Pno_filled(COMPACT, 2000.0)
        assert got == 2000.0
        assert "I2-9a" in note

    def test_rectangular_Fcr_is_inverse_square(self) -> None:
        """Eq. I2-10: Fcr = 9*Es/(b/t)^2."""
        assert I.Fcr_filled_rectangular(ES, 60.0) == pytest.approx(
            9.0 * ES / 3600.0, rel=1e-12
        )

    def test_round_Fcr_is_a_fifth_root_law(self) -> None:
        """Eq. I2-11's 0.2 exponent makes a round tube far less slenderness-
        sensitive than Eq. I2-10's inverse square -- which is why Table I1.1a
        permits round filled sections to be much more slender."""
        a = I.Fcr_filled_round(50.0, 100.0, ES)
        b = I.Fcr_filled_round(50.0, 200.0, ES)
        assert a / b == pytest.approx(math.pow(2.0, 0.2), rel=1e-12)
        # the rectangular law would have quartered it
        rect_a = I.Fcr_filled_rectangular(ES, 100.0)
        rect_b = I.Fcr_filled_rectangular(ES, 200.0)
        assert rect_a / rect_b == pytest.approx(4.0, rel=1e-12)


# ===========================================================================
# Sect. I3
# ===========================================================================
class TestCompositeFlexure:
    def test_the_lowest_of_the_three_horizontal_shears_governs(self) -> None:
        """Eqs. I3-1a/b/c."""
        value, note = I.horizontal_shear_positive(4.0, 300.0, 50.0, 14.7, 900.0)
        assert value == pytest.approx(min(0.85 * 4 * 300, 50 * 14.7, 900.0), rel=1e-12)
        assert "I3-1b" in note

    def test_anchor_control_means_partial_composite(self) -> None:
        """When Eq. I3-1c governs the beam is only partially composite -- which
        changes how its flexural strength must be computed."""
        _, note = I.horizontal_shear_positive(4.0, 300.0, 50.0, 14.7, 400.0)
        assert "PARTIALLY COMPOSITE" in note

    def test_negative_moment_has_no_concrete_term(self) -> None:
        """Eqs. I3-2a/b: the slab is in tension and Sect. I1.2 neglects it."""
        value, note = I.horizontal_shear_negative(60.0, 3.0, 400.0)
        assert value == pytest.approx(180.0, rel=1e-12)
        assert "I3-2a" in note

    def test_the_flexural_interpolation_is_LINEAR(self) -> None:
        """Eq. I3-3b is linear where Eq. I2-9c for the SAME section in
        compression is quadratic. One member, two different rules."""
        result = I.filled_flexural_strength(
            NONCOMPACT, 2000.0, 1600.0, lam=15.0, lam_p=10.0, lam_r=20.0
        )
        assert result.nominal == pytest.approx(2000.0 - 400.0 * 0.5, rel=1e-12)
        assert "LINEAR" in result.governing.note

    def test_the_two_rules_disagree_across_the_band(self) -> None:
        compression, _ = I.Pno_filled(
            NONCOMPACT, 2000.0, 1600.0, lam=15.0, lam_p=10.0, lam_r=20.0
        )
        flexure = I.filled_flexural_strength(
            NONCOMPACT, 2000.0, 1600.0, lam=15.0, lam_p=10.0, lam_r=20.0
        ).nominal
        assert compression > flexure

    def test_slender_needs_Mcr(self) -> None:
        with pytest.raises(AISC360Error, match="0.70"):
            I.filled_flexural_strength(SLENDER, 2000.0)


# ===========================================================================
# Sect. I5
# ===========================================================================
class TestCompositeInteraction:
    def test_csr_ratio(self) -> None:
        """Eq. I5-2: csr = (As*Fy + Asr*Fyr)/(Ac*f'c)."""
        assert I.csr_ratio(15.0, 50.0, 4.0, 60.0, 200.0, 5.0) == pytest.approx(
            (15 * 50 + 4 * 60) / (200 * 5), rel=1e-12
        )

    def test_round_coefficients_differ_from_rectangular(self) -> None:
        """Table I5.1 gives 0.27 and 1.10/0.95 for round HSS against 0.17 and
        1.06/0.90 for rectangular."""
        rect_cp, _ = I.cp_cm_coefficients(0.8, round_section=False)
        round_cp, _ = I.cp_cm_coefficients(0.8, round_section=True)
        assert round_cp / rect_cp == pytest.approx(0.27 / 0.17, rel=1e-12)

    def test_the_cm_bounds_run_in_opposite_directions(self) -> None:
        """Above csr = 0.5 the bound is a FLOOR of 1.0; below it a CEILING of
        1.67. They are not one expression in two ranges."""
        _, cm_high = I.cp_cm_coefficients(1.5)
        _, cm_low = I.cp_cm_coefficients(0.05)
        assert cm_high >= 1.0
        assert cm_low <= 1.67

    def test_the_branch_point_is_cp_not_0_2(self) -> None:
        """Unlike Eq. H1-1a's fixed 0.2, Eq. I5-1's branch point depends on the
        section through csr -- two filled columns switch at different loads."""
        cp, cm = I.cp_cm_coefficients(0.8)
        above = I.composite_interaction(cp * 1.01 * 100.0, 100.0, 50.0, 100.0, cp, cm)
        below = I.composite_interaction(cp * 0.99 * 100.0, 100.0, 50.0, 100.0, cp, cm)
        assert above.citation.equation == "I5-1a"
        assert below.citation.equation == "I5-1b"

    def test_a_cm_above_one_makes_the_axial_term_negative(self) -> None:
        """Deliberate: modest compression closes concrete cracks and raises the
        flexural capacity of a filled section, so a small Pr genuinely REDUCES
        the interaction ratio."""
        cp, cm = I.cp_cm_coefficients(1.5)
        assert cm > 1.0
        no_axial = I.composite_interaction(0.0, 100.0, 50.0, 100.0, cp, cm)
        some_axial = I.composite_interaction(cp * 0.5 * 100.0, 100.0, 50.0, 100.0, cp, cm)
        assert some_axial.ratio < no_axial.ratio
        assert "NEGATIVE" in some_axial.note

    def test_the_ratio_is_floored_at_zero(self) -> None:
        cp, cm = I.cp_cm_coefficients(1.5)
        assert I.composite_interaction(0.0, 100.0, 0.0, 100.0, cp, cm).ratio >= 0.0


# ===========================================================================
# Sect. I6
# ===========================================================================
class TestLoadTransfer:
    def test_the_two_force_allocations_are_complements(self) -> None:
        """Eq. I6-1 transfers the CONCRETE's share; Eq. I6-2a the STEEL's. The
        two must sum to the applied force."""
        Pr, Fy, As, Pno = 1000.0, 50.0, 15.0, 2000.0
        to_concrete = I.force_to_concrete(Pr, Fy, As, Pno)
        to_steel = I.force_to_steel(Pr, Fy, As, Pno)
        assert to_concrete + to_steel == pytest.approx(Pr, rel=1e-12)

    def test_slender_filled_uses_Fcr_not_Fy(self) -> None:
        """Eq. I6-2b: a slender tube cannot reach Fy before buckling locally."""
        Fcr = I.Fcr_filled_rectangular(ES, 80.0)
        assert I.force_to_steel(1000.0, Fcr, 15.0, 2000.0) < I.force_to_steel(
            1000.0, 50.0, 15.0, 2000.0
        )

    def test_direct_bearing_is_1_7_not_J8s_0_85(self) -> None:
        """Eq. I6-3's 1.7 is exactly twice Sect. J8's 0.85 -- the concrete is
        confined by the section, so it carries the full sqrt(A2/A1) = 2 bonus
        that Eq. J8-2 caps at."""
        composite = I.direct_bearing_strength(4.0, 100.0)
        j8_full_area = J.concrete_bearing_strength(4.0, 100.0)
        assert composite.nominal == pytest.approx(2.0 * j8_full_area.nominal, rel=1e-12)
        assert composite.nominal == pytest.approx(1.7 * 4.0 * 100.0, rel=1e-12)

    def test_bearing_shares_J8s_factors(self) -> None:
        result = I.direct_bearing_strength(4.0, 100.0)
        assert (result.phi, result.omega) == (0.65, 2.31)

    def test_shear_connection_sums_available_strengths(self) -> None:
        """Eq. I6-4 sums ALREADY-FACTORED anchor strengths -- the one place in
        this library where no further phi applies."""
        value, note = I.shear_connection_strength(450.0)
        assert value == 450.0
        assert "already factored" in note

    def test_bond_is_the_lowest_phi_in_the_specification(self) -> None:
        """Sect. I6.3c: phi = 0.50 / Omega = 3.00."""
        assert (I.PHI_BOND, I.OMEGA_BOND) == (0.50, 3.00)
        assert pytest.approx(1.5, rel=1e-12) == I.PHI_BOND * I.OMEGA_BOND
        result = I.bond_strength(24.0, 30.0, 0.08)
        assert "lowest in the Specification" in result.governing.note

    def test_bond_stress_caps_bite_for_real_sections(self) -> None:
        """Rectangular: 12*t/H^2 <= 0.1 ksi. A 6-in. tube with a 1/4-in. wall
        gives 0.083 -- just under; anything smaller or thicker is capped."""
        assert I.bond_stress(0.25, 6.0) == pytest.approx(12 * 0.25 / 36.0, rel=1e-12)
        assert I.bond_stress(0.5, 4.0) == 0.1
        assert I.bond_stress(0.5, 4.0, round_section=True) == 0.2

    def test_the_round_bond_stress_is_twice_the_rectangular_cap(self) -> None:
        assert I.bond_stress(1.0, 2.0, round_section=True) == 2.0 * I.bond_stress(
            1.0, 2.0, round_section=False
        )


# ===========================================================================
# Sect. I8 -- anchors
# ===========================================================================
class TestAnchorReductionFactors:
    @pytest.mark.parametrize(
        ("orientation", "kwargs", "Rg", "Rp"),
        [
            (I.DeckOrientation.NONE, {}, 1.0, 0.75),
            (I.DeckOrientation.PARALLEL, dict(rib_width_to_depth=2.0), 1.0, 0.75),
            (I.DeckOrientation.PARALLEL, dict(rib_width_to_depth=1.2), 0.85, 0.75),
            (I.DeckOrientation.PERPENDICULAR,
             dict(anchors_per_rib=1, emid_ht_at_least_2in=False), 1.0, 0.6),
            (I.DeckOrientation.PERPENDICULAR,
             dict(anchors_per_rib=2, emid_ht_at_least_2in=False), 0.85, 0.6),
            (I.DeckOrientation.PERPENDICULAR,
             dict(anchors_per_rib=3, emid_ht_at_least_2in=False), 0.7, 0.6),
            (I.DeckOrientation.PERPENDICULAR,
             dict(anchors_per_rib=5, emid_ht_at_least_2in=False), 0.7, 0.6),
        ],
    )
    def test_published_values(
        self, orientation: I.DeckOrientation, kwargs: dict[str, Any], Rg: float, Rp: float
    ) -> None:
        """Source: Sect. I8.2a, pp. 16.1-105 to 16.1-106 -- printed constants."""
        got_Rg, got_Rp, _ = I.Rg_Rp(orientation, **kwargs)
        assert (got_Rg, got_Rp) == (Rg, Rp)

    def test_the_emid_ht_detail_is_worth_25_percent(self) -> None:
        """Rp rises from 0.6 to 0.75 when emid-ht >= 2 in. -- the single most
        commonly missed input in Sect. I8."""
        _, low, _ = I.Rg_Rp(I.DeckOrientation.PERPENDICULAR, emid_ht_at_least_2in=False)
        _, high, _ = I.Rg_Rp(I.DeckOrientation.PERPENDICULAR, emid_ht_at_least_2in=True)
        assert high / low == pytest.approx(1.25, rel=1e-12)

    def test_crowding_a_rib_is_doubly_penalised(self) -> None:
        """Three anchors per perpendicular rib give Rg*Rp = 0.42 against 0.75
        with no deck -- a 44% loss per anchor."""
        Rg_none, Rp_none, _ = I.Rg_Rp(I.DeckOrientation.NONE)
        Rg_3, Rp_3, _ = I.Rg_Rp(
            I.DeckOrientation.PERPENDICULAR, anchors_per_rib=3, emid_ht_at_least_2in=False
        )
        assert Rg_none * Rp_none == pytest.approx(0.75, rel=1e-12)
        assert Rg_3 * Rp_3 == pytest.approx(0.42, rel=1e-12)
        assert 1.0 - (Rg_3 * Rp_3) / (Rg_none * Rp_none) == pytest.approx(0.44, abs=0.005)


class TestAnchorStrength:
    def test_the_cap_usually_governs_in_normal_concrete(self) -> None:
        """Eq. I8-1: the concrete term is 0.5*Asa*sqrt(f'c*Ec); the cap is
        Rg*Rp*Asa*Fu. In 4 ksi normal-weight concrete the cap governs, so the
        DECK DETAIL sets the strength, not the concrete grade."""
        Asa = math.pi * 0.75**2 / 4.0
        Ec = I.concrete_modulus(145.0, 4.0)
        Qn, note = I.stud_anchor_shear_in_beam(Asa, 4.0, Ec, 1.0, 0.75, 65.0)
        assert "stud shear governs" in note
        assert Qn == pytest.approx(1.0 * 0.75 * Asa * 65.0, rel=1e-12)

    def test_weak_concrete_lets_crushing_govern(self) -> None:
        Asa = math.pi * 0.75**2 / 4.0
        Ec = I.concrete_modulus(110.0, 3.0)
        _, note = I.stud_anchor_shear_in_beam(Asa, 3.0, Ec, 1.0, 0.75, 65.0)
        assert "concrete crushing" in note

    def test_channel_anchor_uses_half_the_web(self) -> None:
        """Eq. I8-2: 0.3*(tf + 0.5*tw)*la*sqrt(f'c*Ec) -- HALF the web, not the
        full thickness, and no Rg*Rp and no steel cap."""
        Ec = I.concrete_modulus(145.0, 4.0)
        got = I.channel_anchor_shear(0.5, 0.3, 6.0, 4.0, Ec)
        assert got == pytest.approx(0.3 * (0.5 + 0.15) * 6.0 * math.sqrt(4.0 * Ec), rel=1e-12)

    def test_shear_and_tension_have_the_same_nominal_but_different_factors(self) -> None:
        """Eqs. I8-3 and I8-4 are both Fu*Asa, but 0.65/2.31 against 0.75/2.00 --
        a stud has MORE available strength in tension than in shear, the
        opposite of the bolt intuition."""
        Asa = 0.442
        shear = I.stud_shear_strength(65.0, Asa)
        tension = I.stud_tensile_strength(65.0, Asa)
        assert shear.nominal == pytest.approx(tension.nominal, rel=1e-12)
        assert tension.available > shear.available
        assert (shear.phi, tension.phi) == (0.65, 0.75)

    def test_breakout_is_out_of_scope(self) -> None:
        """Sect. I8.3a's Eq. I8-3 applies only where concrete breakout is not a
        limit state; breakout itself is an ACI 318 calculation."""
        assert "ACI 318" in I.stud_shear_strength(65.0, 0.442).governing.note


class TestAnchorInteraction:
    def test_the_5_3_exponent(self) -> None:
        """Eq. I8-5 -- unique in the Specification. Everything else uses linear
        terms, squares, or the 8/9 of Eq. H1-1a."""
        result = I.anchor_interaction(6.0, 10.0, 6.0, 10.0)
        assert result.ratio == pytest.approx(2.0 * math.pow(0.6, 5.0 / 3.0), rel=1e-12)

    def test_it_bulges_outward_from_both_the_linear_and_circular_rules(self) -> None:
        """Two effects at 60%: 5/3 gives 0.85, linear 1.20, circular 0.72. Linear
        is needlessly conservative; circular is unconservative."""
        result = I.anchor_interaction(6.0, 10.0, 6.0, 10.0)
        assert result.ratio == pytest.approx(0.85, abs=0.01)
        assert pytest.approx(1.20, rel=1e-12) == 0.6 + 0.6
        assert pytest.approx(0.72, rel=1e-12) == 0.6**2 + 0.6**2
        assert result.is_adequate

    def test_pure_shear_or_pure_tension_reaches_unity_at_capacity(self) -> None:
        assert I.anchor_interaction(10.0, 10.0, 0.0, 10.0).ratio == pytest.approx(
            1.0, rel=1e-12
        )
        assert I.anchor_interaction(0.0, 10.0, 10.0, 10.0).ratio == pytest.approx(
            1.0, rel=1e-12
        )


# ===========================================================================
# Provenance
# ===========================================================================
_EXAMPLES = Path(__file__).parent / "design_examples" / "chapter_i.json"


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
    return [case for case in data.get("cases", []) if not case.get("skip")]


class TestProvenance:
    @pytest.mark.design_example
    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: str(c.get("id", "?")))
    def test_case(self, case: dict[str, Any]) -> None:
        kind = case["kind"]
        tol = float(case.get("tol", 0.005))

        if kind == "Rg_Rp":
            Rg, Rp, _ = I.Rg_Rp(
                I.DeckOrientation(case["orientation"]),
                anchors_per_rib=int(case.get("anchors_per_rib", 1)),
                rib_width_to_depth=float(case.get("rib_width_to_depth", 2.0)),
                emid_ht_at_least_2in=bool(case.get("emid_ht_at_least_2in", True)),
            )
            assert [Rg, Rp] == pytest.approx(case["expected"], abs=1e-12)
        elif kind == "Ec":
            got = I.concrete_modulus(float(case["wc"]), float(case["fc_prime"]))
            assert got == pytest.approx(float(case["expected"]), rel=tol)
        elif kind == "Pno_encased":
            got = I.Pno_encased(*[float(x) for x in case["args"]])
            assert got == pytest.approx(float(case["expected"]), rel=tol)
        else:  # pragma: no cover
            raise AssertionError(f"unknown case kind {kind!r}")

    def test_every_case_declares_its_source(self) -> None:
        for case in _load_cases():
            assert case.get("source"), f"case {case.get('id')!r} has no source"
