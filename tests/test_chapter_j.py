"""Chapter J -- Design of Connections.

Chapter J uses six different phi/Omega pairs, sometimes two inside one section,
so the tests lead with the factors: :class:`TestFactorDiversity` pins each pair
and :class:`TestOrderingInversions` shows the cases where comparing nominal
strengths would pick the wrong limit state.

Precision: 1e-12 for the algebra and the published table constants, which are
exact; 0.5% for anything traceable to a rounded catalog figure.
"""

from __future__ import annotations

import math

import pytest

from pyaisc360 import chapter_j as J
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import LimitState
from pyaisc360.core.exceptions import AISC360Error, GeometryError, OutOfScopeError

E_STEEL = 29000.0


# ===========================================================================
# Factors
# ===========================================================================
class TestFactorDiversity:
    """Chapter J's defining feature: six different factor pairs."""

    @pytest.mark.parametrize(
        ("phi", "omega", "where"),
        [
            (J.PHI_WELD, J.OMEGA_WELD, "J2.4 weld metal"),
            (J.PHI_WELD_PJP_TENSION, J.OMEGA_WELD_PJP_TENSION, "Table J2.5 PJP tension"),
            (J.PHI_BOLT, J.OMEGA_BOLT, "J3.6 bolts"),
            (J.PHI_YIELDING, J.OMEGA_YIELDING, "J4.1(a) tensile yielding"),
            (J.PHI_RUPTURE, J.OMEGA_RUPTURE, "J4.1(b) tensile rupture"),
            (J.PHI_SHEAR_YIELDING, J.OMEGA_SHEAR_YIELDING, "J4.2(a) shear yielding"),
            (J.PHI_CONCRETE, J.OMEGA_CONCRETE, "J8 concrete"),
        ],
    )
    def test_every_pair_is_calibrated(self, phi: float, omega: float, where: str) -> None:
        """Omega = 1.5/phi holds for all of them, including the odd 0.65/2.31."""
        assert phi * omega == pytest.approx(1.5, abs=0.01), where

    def test_the_three_phi_equals_one_cases(self) -> None:
        """phi = 1.00 appears only for shear yielding (J4.2(a), J10.2) and slip
        in standard holes (J3.8) -- plus Sect. G2.1(a) outside this chapter."""
        assert J.PHI_SHEAR_YIELDING == 1.00
        assert J.slip_factors(J.HoleType.STANDARD)[0] == 1.00

    def test_concrete_bearing_has_its_own_pair(self) -> None:
        """Sect. J8's 0.65/2.31 is unique in the Specification -- the limit state
        is crushing of concrete, not of steel."""
        assert (J.PHI_CONCRETE, J.OMEGA_CONCRETE) == (0.65, 2.31)


# ===========================================================================
# Tables J3.1 and J3.2
# ===========================================================================
class TestTableJ3_2:
    @pytest.mark.parametrize(
        ("group", "threads", "Fnt", "Fnv"),
        [
            (J.BoltGroup.A307, J.ThreadCondition.INCLUDED, 45.0, 27.0),
            (J.BoltGroup.GROUP_A, J.ThreadCondition.INCLUDED, 90.0, 54.0),
            (J.BoltGroup.GROUP_A, J.ThreadCondition.EXCLUDED, 90.0, 68.0),
            (J.BoltGroup.GROUP_B, J.ThreadCondition.INCLUDED, 113.0, 68.0),
            (J.BoltGroup.GROUP_B, J.ThreadCondition.EXCLUDED, 113.0, 84.0),
            (J.BoltGroup.GROUP_C, J.ThreadCondition.INCLUDED, 150.0, 90.0),
            (J.BoltGroup.GROUP_C, J.ThreadCondition.EXCLUDED, 150.0, 113.0),
        ],
    )
    def test_published_values(
        self, group: J.BoltGroup, threads: J.ThreadCondition, Fnt: float, Fnv: float
    ) -> None:
        """Source: Table J3.2, p. 16.1-129 -- printed constants, asserted exactly."""
        assert J.nominal_bolt_stress(group, threads) == (Fnt, Fnv)

    def test_threads_change_Fnv_but_never_Fnt(self) -> None:
        """A bolt in tension is stressed through its threads either way."""
        for group in (J.BoltGroup.GROUP_A, J.BoltGroup.GROUP_B, J.BoltGroup.GROUP_C):
            included = J.nominal_bolt_stress(group, J.ThreadCondition.INCLUDED)
            excluded = J.nominal_bolt_stress(group, J.ThreadCondition.EXCLUDED)
            assert included[0] == excluded[0]
            assert excluded[1] > included[1]

    def test_excluding_threads_gains_about_25_percent(self) -> None:
        for group in (J.BoltGroup.GROUP_A, J.BoltGroup.GROUP_B):
            included = J.nominal_bolt_stress(group, J.ThreadCondition.INCLUDED)[1]
            excluded = J.nominal_bolt_stress(group, J.ThreadCondition.EXCLUDED)[1]
            assert 1.2 < excluded / included < 1.3

    def test_A307_permits_threads_in_shear_planes(self) -> None:
        """Footnote [d]: both conditions give the same Fnv for A307."""
        assert J.nominal_bolt_stress(J.BoltGroup.A307, J.ThreadCondition.INCLUDED) == (
            J.nominal_bolt_stress(J.BoltGroup.A307, J.ThreadCondition.EXCLUDED)
        )

    def test_threaded_parts_scale_with_Fu(self) -> None:
        """0.75*Fu tensile, 0.450*Fu or 0.563*Fu shear."""
        Fnt, Fnv = J.nominal_bolt_stress(
            J.BoltGroup.THREADED_PART, J.ThreadCondition.INCLUDED, Fu=58.0
        )
        assert Fnt == pytest.approx(0.75 * 58.0, rel=1e-12)
        assert Fnv == pytest.approx(0.450 * 58.0, rel=1e-12)

    def test_threaded_parts_need_Fu(self) -> None:
        with pytest.raises(AISC360Error, match="need Fu"):
            J.nominal_bolt_stress(J.BoltGroup.THREADED_PART, J.ThreadCondition.INCLUDED)


class TestTableJ3_1:
    @pytest.mark.parametrize(
        ("diameter", "group", "Tb"),
        [
            (0.750, J.BoltGroup.GROUP_A, 28.0),
            (0.875, J.BoltGroup.GROUP_A, 39.0),
            (1.000, J.BoltGroup.GROUP_A, 51.0),
            (0.750, J.BoltGroup.GROUP_B, 35.0),
            (1.000, J.BoltGroup.GROUP_B, 64.0),
            (1.125, J.BoltGroup.GROUP_C, 113.0),
        ],
    )
    def test_published_pretensions(
        self, diameter: float, group: J.BoltGroup, Tb: float
    ) -> None:
        """Source: Table J3.1, p. 16.1-127."""
        assert J.minimum_pretension(group, diameter) == Tb

    def test_group_B_always_exceeds_group_A(self) -> None:
        for diameter in (0.5, 0.75, 1.0, 1.25, 1.5):
            a = J.minimum_pretension(J.BoltGroup.GROUP_A, diameter)
            b = J.minimum_pretension(J.BoltGroup.GROUP_B, diameter)
            assert b > a

    def test_group_C_is_only_tabulated_for_three_diameters(self) -> None:
        for diameter in (1.000, 1.125, 1.250):
            assert J.minimum_pretension(J.BoltGroup.GROUP_C, diameter) > 0.0
        with pytest.raises(AISC360Error, match="no pretension"):
            J.minimum_pretension(J.BoltGroup.GROUP_C, 0.750)

    def test_unlisted_diameter_raises(self) -> None:
        with pytest.raises(AISC360Error, match="does not list"):
            J.minimum_pretension(J.BoltGroup.GROUP_A, 0.6875)


# ===========================================================================
# Sect. J2 -- welds
# ===========================================================================
class TestWeldLength:
    def test_short_welds_are_fully_effective(self) -> None:
        length, note = J.fillet_weld_effective_length(20.0, 0.25)  # l/w = 80
        assert length == 20.0
        assert "100w" in note

    def test_beta_reduces_a_long_weld(self) -> None:
        """Eq. J2-1: beta = 1.2 - 0.002*(l/w). At l/w = 150, beta = 0.90."""
        assert J.weld_length_reduction(37.5, 0.25) == pytest.approx(0.90, rel=1e-12)

    def test_beta_is_capped_at_unity(self) -> None:
        assert J.weld_length_reduction(10.0, 0.25) == 1.0

    def test_the_180w_cap_meets_beta_exactly_at_300w(self) -> None:
        """At l = 300w, Eq. J2-1 gives beta = 0.6 and beta*l = 180w. The two
        branches meet exactly -- beyond that, more weld adds nothing."""
        w = 0.25
        at_limit, _ = J.fillet_weld_effective_length(300.0 * w, w)
        just_past, note = J.fillet_weld_effective_length(300.0 * w * 1.0001, w)
        assert at_limit == pytest.approx(180.0 * w, rel=1e-9)
        assert just_past == pytest.approx(180.0 * w, rel=1e-12)
        assert "adds nothing" in note

    def test_effective_length_never_increases_past_300w(self) -> None:
        w = 0.25
        lengths = [J.fillet_weld_effective_length(x * w, w)[0] for x in (300, 400, 800)]
        assert all(x == pytest.approx(180.0 * w, rel=1e-9) for x in lengths)


class TestWeldStrength:
    def test_longitudinal_fillet_has_no_increase(self) -> None:
        """theta = 0: sin(0) = 0, so the factor is exactly 1.0."""
        assert J.directional_strength_increase(0.0) == pytest.approx(1.0, rel=1e-12)

    def test_transverse_fillet_gains_50_percent(self) -> None:
        """theta = 90: sin^1.5(90) = 1, so the factor is exactly 1.5."""
        assert J.directional_strength_increase(90.0) == pytest.approx(1.5, rel=1e-12)

    def test_intermediate_angle(self) -> None:
        assert J.directional_strength_increase(45.0) == pytest.approx(
            1.0 + 0.5 * math.sin(math.radians(45.0)) ** 1.5, rel=1e-12
        )

    def test_increase_is_off_by_default(self) -> None:
        """Sect. J2.4(b) permits it only if strain compatibility is considered."""
        plain = J.fillet_weld_strength(70.0, 1.0)
        assert plain.nominal == pytest.approx(0.60 * 70.0, rel=1e-12)
        assert "no directional increase" in plain.governing.note

    def test_E70_fillet_nominal_stress(self) -> None:
        """Table J2.5: Fnw = 0.60*FEXX = 42 ksi for E70XX."""
        assert J.fillet_weld_strength(70.0, 1.0).nominal == pytest.approx(42.0, rel=1e-12)

    def test_weld_factors(self) -> None:
        result = J.fillet_weld_strength(70.0, 1.0)
        assert (result.phi, result.omega) == (0.75, 2.00)
        assert result.available == pytest.approx(0.75 * 42.0, rel=1e-12)

    def test_theta_outside_range_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match=r"\[0, 90\]"):
            J.directional_strength_increase(120.0)


class TestWeldGroup:
    def test_J2_6a_governs_when_longitudinal_dominates(self) -> None:
        Rn, note = J.weld_group_strength(Rnwl=100.0, Rnwt=10.0)
        assert Rn == pytest.approx(110.0, rel=1e-12)
        assert "J2-6a" in note

    def test_J2_6b_governs_when_transverse_dominates(self) -> None:
        Rn, note = J.weld_group_strength(Rnwl=10.0, Rnwt=100.0)
        assert Rn == pytest.approx(0.85 * 10.0 + 1.5 * 100.0, rel=1e-12)
        assert "J2-6b" in note

    def test_the_greater_of_the_two_is_always_taken(self) -> None:
        for l, t in [(100.0, 10.0), (50.0, 50.0), (10.0, 100.0), (0.0, 100.0)]:
            Rn, _ = J.weld_group_strength(l, t)
            assert Rn == pytest.approx(max(l + t, 0.85 * l + 1.5 * t), rel=1e-12)

    def test_the_crossover(self) -> None:
        """0.85*L + 1.5*T = L + T at T = 0.3*L."""
        Rn_a, _ = J.weld_group_strength(100.0, 30.0)
        assert Rn_a == pytest.approx(130.0, rel=1e-12)
        assert pytest.approx(130.0, rel=1e-12) == 0.85 * 100.0 + 1.5 * 30.0


# ===========================================================================
# Sect. J3 -- bolts
# ===========================================================================
class TestBoltStrength:
    def test_uses_the_nominal_body_area(self) -> None:
        """Eq. J3-1: Ab is the UNTHREADED body area. The thread reduction is
        already inside Fn, so using the tensile stress area double-counts it."""
        Ab = math.pi * 0.75**2 / 4.0
        result = J.bolt_strength(54.0, Ab)
        assert result.nominal == pytest.approx(54.0 * Ab, rel=1e-12)
        assert "UNTHREADED" in result.governing.note

    def test_A325_3_4_inch_single_shear(self) -> None:
        """3/4 in. Group A, threads included: Fnv = 54 ksi, Ab = 0.4418 in^2."""
        Ab = math.pi * 0.75**2 / 4.0
        _, Fnv = J.nominal_bolt_stress(J.BoltGroup.GROUP_A, J.ThreadCondition.INCLUDED)
        result = J.bolt_strength(Fnv, Ab)
        assert result.available == pytest.approx(0.75 * 54.0 * Ab, rel=1e-12)
        assert result.available == pytest.approx(17.9, rel=0.005)

    def test_prying_is_the_callers_responsibility(self) -> None:
        assert "prying" in J.bolt_strength(90.0, 0.4418).governing.note


class TestCombinedTensionShear:
    def test_small_shear_costs_nothing(self) -> None:
        """The 1.3 multiplier means Eq. J3-3a starts above Fnt, then caps at Fnt.

        The User Note's 30% rule falls out of this: below that the cap binds.
        """
        Fnt, Fnv = 90.0, 54.0
        modified, note = J.combined_tension_shear(Fnt, Fnv, frv=0.3 * 0.75 * Fnv)
        assert modified == pytest.approx(Fnt, rel=1e-12)
        assert "cost nothing" in note

    def test_large_shear_reduces_the_tensile_stress(self) -> None:
        Fnt, Fnv = 90.0, 54.0
        frv = 0.9 * 0.75 * Fnv
        modified, _ = J.combined_tension_shear(Fnt, Fnv, frv)
        assert modified < Fnt
        assert modified == pytest.approx(1.3 * Fnt - (Fnt / (0.75 * Fnv)) * frv, rel=1e-12)

    def test_asd_form_uses_omega(self) -> None:
        Fnt, Fnv, frv = 90.0, 54.0, 30.0
        lrfd, _ = J.combined_tension_shear(Fnt, Fnv, frv, Basis.LRFD)
        asd, _ = J.combined_tension_shear(Fnt, Fnv, frv, Basis.ASD)
        assert asd == pytest.approx(min(1.3 * Fnt - (2.0 * Fnt / Fnv) * frv, Fnt), rel=1e-12)
        assert asd < lrfd

    def test_result_is_floored_at_zero(self) -> None:
        modified, _ = J.combined_tension_shear(90.0, 54.0, frv=54.0)
        assert modified >= 0.0


class TestSlipCritical:
    def test_slip_resistance(self) -> None:
        """Eq. J3-4: Rn = mu*Du*hf*Tb*ns."""
        result = J.slip_resistance(0.30, 1.13, 1.0, 39.0, 1, J.HoleType.STANDARD)
        assert result.nominal == pytest.approx(0.30 * 1.13 * 1.0 * 39.0 * 1, rel=1e-12)

    def test_class_B_is_two_thirds_stronger(self) -> None:
        """mu = 0.50 against 0.30."""
        assert J.FayingSurface.CLASS_A.mu == 0.30
        assert J.FayingSurface.CLASS_B.mu == 0.50
        a = J.slip_resistance(0.30, 1.13, 1.0, 39.0, 1, J.HoleType.STANDARD).nominal
        b = J.slip_resistance(0.50, 1.13, 1.0, 39.0, 1, J.HoleType.STANDARD).nominal
        assert b / a == pytest.approx(5.0 / 3.0, rel=1e-12)

    @pytest.mark.parametrize(
        ("hole", "phi", "omega"),
        [
            (J.HoleType.STANDARD, 1.00, 1.50),
            (J.HoleType.SHORT_SLOT_PERPENDICULAR, 1.00, 1.50),
            (J.HoleType.OVERSIZE, 0.85, 1.76),
            (J.HoleType.SHORT_SLOT_PARALLEL, 0.85, 1.76),
            (J.HoleType.LONG_SLOT_PARALLEL, 0.70, 2.14),
        ],
    )
    def test_slip_factors_by_hole_type(
        self, hole: J.HoleType, phi: float, omega: float
    ) -> None:
        """Sect. J3.8, p. 16.1-134 -- published constants."""
        assert J.slip_factors(hole) == (phi, omega)

    def test_pretension_uses_Tb_not_the_applied_force(self) -> None:
        """Eq. J3-4 takes Tb, the MINIMUM pretension from Table J3.1."""
        Tb = J.minimum_pretension(J.BoltGroup.GROUP_A, 0.875)
        result = J.slip_resistance(0.30, 1.13, 1.0, Tb, 2, J.HoleType.STANDARD)
        assert result.governing.detail["Tb"] == 39.0

    def test_two_slip_planes_double_the_resistance(self) -> None:
        one = J.slip_resistance(0.30, 1.13, 1.0, 39.0, 1, J.HoleType.STANDARD).nominal
        two = J.slip_resistance(0.30, 1.13, 1.0, 39.0, 2, J.HoleType.STANDARD).nominal
        assert two == pytest.approx(2.0 * one, rel=1e-12)


class TestSlipTensionReduction:
    def test_no_tension_gives_unity(self) -> None:
        assert J.slip_tension_reduction(0.0, 1.13, 39.0, 4, Basis.LRFD) == 1.0

    def test_tension_reduces_the_clamping(self) -> None:
        """Eq. J3-5a: ksc = 1 - Tu/(Du*Tb*nb)."""
        ksc = J.slip_tension_reduction(40.0, 1.13, 39.0, 4, Basis.LRFD)
        assert ksc == pytest.approx(1.0 - 40.0 / (1.13 * 39.0 * 4), rel=1e-12)

    def test_asd_uses_1_5_not_1_6(self) -> None:
        """Sect. J3.9 states 1.5 explicitly -- not the 1.6 used for ASD force
        level elsewhere in the Specification."""
        lrfd = J.slip_tension_reduction(40.0, 1.13, 39.0, 4, Basis.LRFD)
        asd = J.slip_tension_reduction(40.0, 1.13, 39.0, 4, Basis.ASD)
        assert (1.0 - asd) / (1.0 - lrfd) == pytest.approx(1.5, rel=1e-12)

    def test_floored_at_zero(self) -> None:
        """Once the applied tension reaches the total pretension, no clamping
        remains -- and a negative slip resistance is meaningless."""
        assert J.slip_tension_reduction(1000.0, 1.13, 39.0, 4, Basis.LRFD) == 0.0


class TestBearingAndTearout:
    def test_bearing_deformation_considered(self) -> None:
        """Eq. J3-6a: Rn = 2.4*d*t*Fu."""
        Rn, note = J.bearing_strength(0.75, 0.5, 58.0)
        assert Rn == pytest.approx(2.4 * 0.75 * 0.5 * 58.0, rel=1e-12)
        assert "J3-6a" in note

    def test_bearing_deformation_not_considered(self) -> None:
        """Eq. J3-6b: 3.0 instead of 2.4 -- a 25% gain, but it permits the hole
        to ovalise beyond 1/4 in., which most connections cannot tolerate."""
        Rn, note = J.bearing_strength(0.75, 0.5, 58.0, deformation_considered=False)
        assert Rn == pytest.approx(3.0 * 0.75 * 0.5 * 58.0, rel=1e-12)
        assert "J3-6b" in note

    def test_deformation_considered_is_the_default(self) -> None:
        assert J.bearing_strength(0.75, 0.5, 58.0)[0] < J.bearing_strength(
            0.75, 0.5, 58.0, deformation_considered=False
        )[0]

    def test_long_slot_perpendicular_is_weakest(self) -> None:
        """Eq. J3-6e: 2.0 instead of 2.4."""
        Rn, note = J.bearing_strength(0.75, 0.5, 58.0, long_slot_perpendicular=True)
        assert Rn == pytest.approx(2.0 * 0.75 * 0.5 * 58.0, rel=1e-12)
        assert "J3-6e" in note

    def test_tearout_uses_the_clear_distance(self) -> None:
        """Eq. J3-6c: Rn = 1.2*lc*t*Fu, with lc the CLEAR distance."""
        Rn, note = J.tearout_strength(1.5, 0.5, 58.0)
        assert Rn == pytest.approx(1.2 * 1.5 * 0.5 * 58.0, rel=1e-12)
        assert "J3-6c" in note

    def test_combined_returns_both_states(self) -> None:
        result = J.bolt_hole_strength(0.75, 1.5, 0.5, 58.0)
        states = {s.limit_state for s in result.limit_states}
        assert states == {LimitState.BOLT_BEARING, LimitState.BOLT_TEAROUT}

    def test_a_short_edge_distance_makes_tearout_govern(self) -> None:
        near_edge = J.bolt_hole_strength(0.75, 0.6, 0.5, 58.0)
        far_edge = J.bolt_hole_strength(0.75, 3.0, 0.5, 58.0)
        assert near_edge.limit_state is LimitState.BOLT_TEAROUT
        assert far_edge.limit_state is LimitState.BOLT_BEARING

    def test_the_tearout_bearing_crossover(self) -> None:
        """1.2*lc = 2.4*d at lc = 2*d -- exactly two diameters of clear distance."""
        d = 0.75
        result = J.bolt_hole_strength(d, 2.0 * d, 0.5, 58.0)
        bearing = next(s for s in result.limit_states if s.limit_state is LimitState.BOLT_BEARING)
        tearout = next(s for s in result.limit_states if s.limit_state is LimitState.BOLT_TEAROUT)
        assert bearing.nominal == pytest.approx(tearout.nominal, rel=1e-12)

    def test_the_clear_distance_warning_is_on_the_result(self) -> None:
        result = J.bolt_hole_strength(0.75, 1.5, 0.5, 58.0)
        tearout = next(s for s in result.limit_states if s.limit_state is LimitState.BOLT_TEAROUT)
        assert "CLEAR distance" in tearout.note


# ===========================================================================
# Sect. J4 -- connecting elements
# ===========================================================================
class TestJ4Elements:
    def test_tension_states_carry_different_factors(self) -> None:
        result = J.element_tension_strength(36.0, 2.5, 58.0, 2.0)
        by_state = {s.limit_state: (s.phi, s.omega) for s in result.limit_states}
        assert by_state[LimitState.TENSILE_YIELDING] == (0.90, 1.67)
        assert by_state[LimitState.TENSILE_RUPTURE] == (0.75, 2.00)

    def test_shear_yielding_has_phi_one(self) -> None:
        """Sect. J4.2(a): Rn = 0.60*Fy*Agv at phi = 1.00."""
        result = J.element_shear_strength(36.0, 6.0, 58.0, 4.0)
        yielding = next(
            s for s in result.limit_states if s.limit_state is LimitState.SHEAR_YIELDING
        )
        assert yielding.nominal == pytest.approx(0.60 * 36.0 * 6.0, rel=1e-12)
        assert yielding.phi == 1.00

    def test_shear_rupture_has_phi_0_75(self) -> None:
        result = J.element_shear_strength(36.0, 6.0, 58.0, 4.0)
        rupture = next(
            s for s in result.limit_states if s.limit_state is LimitState.SHEAR_RUPTURE
        )
        assert rupture.nominal == pytest.approx(0.60 * 58.0 * 4.0, rel=1e-12)
        assert rupture.phi == 0.75

    def test_compression_shortcut_below_Lc_over_r_of_25(self) -> None:
        """Eq. J4-6: Pn = Fy*Ag. At Lc/r = 25 the Chapter E answer is 0.97*Fy,
        so the 3% error is not worth a buckling calculation."""
        result = J.element_compression_strength(36.0, 2.5, 20.0)
        assert result.nominal == pytest.approx(36.0 * 2.5, rel=1e-12)

    def test_above_25_directs_to_chapter_E(self) -> None:
        with pytest.raises(OutOfScopeError, match="Chapter E"):
            J.element_compression_strength(36.0, 2.5, 30.0)

    def test_the_shortcut_is_within_3_percent_of_chapter_E(self) -> None:
        """Verified against Eq. E3-2 rather than asserted."""
        from pyaisc360 import utils

        Fe = utils.elastic_buckling_stress(25.0, E_STEEL)
        Fcr = utils.flexural_buckling_stress(36.0, Fe)
        assert 0.96 < Fcr / 36.0 < 1.0


class TestBlockShear:
    def test_matches_the_published_worked_value(self) -> None:
        """Source: aisc-steel-design/scripts/validate.py, Eq. J4-5.

        Rn = min(0.6*58*4, 0.6*36*6) + 58*1.5 = 129.6 + 87.0 = 216.6 kips
        """
        result = J.block_shear_strength(58.0, 4.0, 1.5, 36.0, 6.0)
        assert result.nominal == pytest.approx(216.6, rel=1e-12)

    def test_the_tension_term_is_identical_on_both_sides(self) -> None:
        """Only the SHEAR term is capped -- Eq. J4-5 is not a comparison of two
        independent expressions. Doubling Ant raises the answer by exactly
        Ubs*Fu*Ant either way."""
        a = J.block_shear_strength(58.0, 4.0, 1.5, 36.0, 6.0).nominal
        b = J.block_shear_strength(58.0, 4.0, 3.0, 36.0, 6.0).nominal
        assert b - a == pytest.approx(58.0 * 1.5, rel=1e-12)

    def test_shear_yielding_caps_the_shear_term(self) -> None:
        """A large Anv relative to Agv cannot help past 0.60*Fy*Agv."""
        result = J.block_shear_strength(58.0, 10.0, 1.5, 36.0, 6.0)
        assert result.nominal == pytest.approx(0.60 * 36.0 * 6.0 + 58.0 * 1.5, rel=1e-12)
        assert "caps the shear term" in result.governing.note

    def test_Ubs_halves_the_tension_term(self) -> None:
        uniform = J.block_shear_strength(58.0, 4.0, 1.5, 36.0, 6.0, Ubs=1.0).nominal
        nonuniform = J.block_shear_strength(58.0, 4.0, 1.5, 36.0, 6.0, Ubs=0.5).nominal
        assert uniform - nonuniform == pytest.approx(0.5 * 58.0 * 1.5, rel=1e-12)

    def test_only_the_two_published_Ubs_values_are_accepted(self) -> None:
        with pytest.raises(AISC360Error, match="Ubs is 1.0"):
            J.block_shear_strength(58.0, 4.0, 1.5, 36.0, 6.0, Ubs=0.75)

    def test_factors(self) -> None:
        result = J.block_shear_strength(58.0, 4.0, 1.5, 36.0, 6.0)
        assert (result.phi, result.omega) == (0.75, 2.00)


class TestOrderingInversions:
    """Where comparing nominal strengths would pick the wrong limit state.

    Chapter J's factor spread makes this common, not exotic -- Sect. J4.2 puts
    shear yielding at phi = 1.00 against shear rupture at 0.75, so the orderings
    disagree whenever ``0.75 < 0.6*Fy*Agv/(0.6*Fu*Anv) < 1.0``.
    """

    def test_J4_2_shear_orderings_disagree(self) -> None:
        """Fy = 36, Agv = 6.0, Fu = 58, Anv = 4.0.

        nominal:   yielding 129.6 < rupture 139.2  -> yielding is lower
        available: yielding 129.6 > rupture 104.4  -> RUPTURE governs
        """
        result = J.element_shear_strength(36.0, 6.0, 58.0, 4.0)
        lowest_nominal = min(result.limit_states, key=lambda s: s.nominal)
        assert lowest_nominal.limit_state is LimitState.SHEAR_YIELDING
        assert result.governing.limit_state is LimitState.SHEAR_RUPTURE
        assert result.available == pytest.approx(0.75 * 0.60 * 58.0 * 4.0, rel=1e-12)

    def test_J4_1_tension_orderings_can_disagree(self) -> None:
        """The same inversion Sect. D2 has, for the same reason."""
        result = J.element_tension_strength(36.0, 2.5, 58.0, 1.7)
        assert 36.0 * 2.5 < 58.0 * 1.7
        assert result.governing.limit_state is LimitState.TENSILE_RUPTURE

    def test_report_shows_both_columns(self) -> None:
        text = J.element_shear_strength(36.0, 6.0, 58.0, 4.0).report()
        assert "1.00" in text and "0.75" in text and "governs" in text


# ===========================================================================
# Sects. J7 and J8
# ===========================================================================
class TestBearingOnSurfaces:
    def test_milled_surface_bearing(self) -> None:
        """Eq. J7-1: Rn = 1.8*Fy*Apb. The 1.8 reflects confined yielding."""
        result = J.surface_bearing_strength(36.0, 10.0)
        assert result.nominal == pytest.approx(1.8 * 36.0 * 10.0, rel=1e-12)

    def test_small_roller(self) -> None:
        """Eq. J7-2: Rn = 1.2*(Fy - 13)*lb*d/20."""
        Rn, note = J.roller_bearing_strength(36.0, 6.0, 10.0)
        assert Rn == pytest.approx(1.2 * 23.0 * 6.0 * 10.0 / 20.0, rel=1e-12)
        assert "J7-2" in note

    def test_large_roller_changes_form(self) -> None:
        """Eq. J7-3 goes as sqrt(d), not d -- a change of form at 25 in., not a
        change of coefficient."""
        Rn, note = J.roller_bearing_strength(36.0, 6.0, 36.0)
        assert Rn == pytest.approx(6.0 * 23.0 * 6.0 * math.sqrt(36.0) / 20.0, rel=1e-12)
        assert "J7-3" in note

    def test_low_yield_stress_is_out_of_scope(self) -> None:
        """Eqs. J7-2/J7-3 subtract 13 ksi and are dimensional, not general."""
        with pytest.raises(OutOfScopeError, match="empirical"):
            J.roller_bearing_strength(12.0, 6.0, 10.0)


class TestConcreteBearing:
    def test_full_area(self) -> None:
        """Eq. J8-1: Pp = 0.85*fc'*A1."""
        result = J.concrete_bearing_strength(4.0, 100.0)
        assert result.nominal == pytest.approx(0.85 * 4.0 * 100.0, rel=1e-12)

    def test_confinement_bonus(self) -> None:
        """Eq. J8-2: sqrt(A2/A1) with A2 = 4*A1 gives a factor of 2."""
        result = J.concrete_bearing_strength(4.0, 100.0, 400.0)
        assert result.nominal == pytest.approx(0.85 * 4.0 * 100.0 * 2.0, rel=1e-12)

    def test_the_bonus_is_capped_at_two(self) -> None:
        """The 1.7*fc'*A1 limit is exactly 2 x the Eq. J8-1 value."""
        result = J.concrete_bearing_strength(4.0, 100.0, 2500.0)
        assert result.nominal == pytest.approx(1.7 * 4.0 * 100.0, rel=1e-12)
        assert "capped" in result.governing.note

    def test_A2_below_A1_falls_back_to_the_full_area_form(self) -> None:
        result = J.concrete_bearing_strength(4.0, 100.0, 80.0)
        assert result.citation.equation == "J8-1"

    def test_concrete_factors(self) -> None:
        result = J.concrete_bearing_strength(4.0, 100.0)
        assert (result.phi, result.omega) == (0.65, 2.31)
        assert result.available == pytest.approx(0.65 * 340.0, rel=1e-12)


# ===========================================================================
# Sect. J10 -- concentrated forces
# ===========================================================================
class TestJ10:
    def test_flange_local_bending_depends_only_on_tf_squared(self) -> None:
        """Eq. J10-1: Rn = 6.25*Fyf*tf^2 -- no bf, no bearing length."""
        a = J.flange_local_bending(50.0, 0.5).nominal
        b = J.flange_local_bending(50.0, 1.0).nominal
        assert b / a == pytest.approx(4.0, rel=1e-12)
        assert a == pytest.approx(6.25 * 50.0 * 0.25, rel=1e-12)

    def test_flange_local_bending_halves_near_the_end(self) -> None:
        full = J.flange_local_bending(50.0, 0.5).nominal
        near = J.flange_local_bending(50.0, 0.5, near_member_end=True).nominal
        assert near == pytest.approx(0.5 * full, rel=1e-12)

    def test_web_local_yielding_branches(self) -> None:
        """Eqs. J10-2 and J10-3: 5k against 2.5k. The lb term is unaffected."""
        interior = J.web_local_yielding(50.0, 0.355, 1.0, 4.0)
        end = J.web_local_yielding(50.0, 0.355, 1.0, 4.0, near_member_end=True)
        assert interior.nominal == pytest.approx(50.0 * 0.355 * (5.0 + 4.0), rel=1e-12)
        assert end.nominal == pytest.approx(50.0 * 0.355 * (2.5 + 4.0), rel=1e-12)

    def test_web_local_yielding_has_phi_one(self) -> None:
        assert J.web_local_yielding(50.0, 0.355, 1.0, 4.0).phi == 1.00

    def test_crippling_end_form_is_half_the_interior_form(self) -> None:
        """Eq. J10-5a is exactly 0.40/0.80 of Eq. J10-4 at the same lb/d."""
        kwargs = dict(tw=0.355, tf=0.570, d=18.0, lb=3.0, E=E_STEEL, Fyw=50.0)
        interior = J.web_local_crippling(**kwargs).nominal
        end = J.web_local_crippling(**kwargs, near_member_end=True).nominal
        assert 3.0 / 18.0 <= 0.2
        assert end == pytest.approx(0.5 * interior, rel=1e-12)

    def test_crippling_J10_5a_and_J10_5b_meet_at_lb_over_d_of_0_2(self) -> None:
        """3*(0.2) = 0.6 and 4*(0.2) - 0.2 = 0.6 -- exactly equal."""
        assert pytest.approx(4.0 * 0.2 - 0.2, rel=1e-12) == 3.0 * 0.2
        d = 18.0
        below = J.web_local_crippling(
            0.355, 0.570, d, 0.2 * d, E_STEEL, 50.0, near_member_end=True
        ).nominal
        above = J.web_local_crippling(
            0.355, 0.570, d, 0.2 * d * (1 + 1e-12), E_STEEL, 50.0, near_member_end=True
        ).nominal
        assert above == pytest.approx(below, rel=1e-9)

    def test_sidesway_buckling_returns_None_above_the_threshold(self) -> None:
        """Sect. J10.4: the limit state "does not apply" -- which is not the same
        as a strength of zero."""
        # (h/tw)/(Lb/bf) = (15.5/0.355)/(Lb/7.5) = 43.66/(Lb/7.5).
        # Lb = 200 -> 1.637 <= 2.3, so the limit state applies.
        applies = J.web_sidesway_buckling(960000.0, 0.355, 0.570, 15.5, 200.0, 7.5)
        assert applies is not None
        # Lb = 30 -> 10.9 > 2.3, so it does not.
        does_not = J.web_sidesway_buckling(960000.0, 0.355, 0.570, 15.5, 30.0, 7.5)
        assert does_not is None

    def test_unrestrained_flange_lacks_the_leading_one(self) -> None:
        """Eq. J10-7 has only the cubic term where Eq. J10-6 has 1 + cubic."""
        kwargs = dict(Cr=960000.0, tw=0.355, tf=0.570, h=15.5, Lb=200.0, bf=7.5)
        restrained = J.web_sidesway_buckling(**kwargs)
        unrestrained = J.web_sidesway_buckling(
            **kwargs, flange_restrained_against_rotation=False
        )
        assert restrained is not None and unrestrained is not None
        assert unrestrained.nominal < restrained.nominal

    def test_compression_buckling(self) -> None:
        """Eq. J10-8: Rn = 24*tw^3*sqrt(E*Fyw)/h."""
        result = J.web_compression_buckling(0.355, 15.5, E_STEEL, 50.0)
        assert result.nominal == pytest.approx(
            24.0 * 0.355**3 * math.sqrt(E_STEEL * 50.0) / 15.5, rel=1e-12
        )

    def test_compression_buckling_halves_near_the_end(self) -> None:
        full = J.web_compression_buckling(0.355, 15.5, E_STEEL, 50.0).nominal
        near = J.web_compression_buckling(
            0.355, 15.5, E_STEEL, 50.0, near_member_end=True
        ).nominal
        assert near == pytest.approx(0.5 * full, rel=1e-12)

    def test_panel_zone_low_axial_load(self) -> None:
        """Eq. J10-9: Rn = 0.60*Fy*dc*tw."""
        result = J.panel_zone_shear(50.0, 14.0, 0.44, 100.0, 1000.0)
        assert result.nominal == pytest.approx(0.60 * 50.0 * 14.0 * 0.44, rel=1e-12)
        assert result.citation.equation == "J10-9"

    def test_panel_zone_high_axial_load_reduces_strength(self) -> None:
        """Eq. J10-10: multiplied by (1.4 - alpha*Pr/Py)."""
        result = J.panel_zone_shear(50.0, 14.0, 0.44, 600.0, 1000.0)
        assert result.citation.equation == "J10-10"
        assert result.nominal == pytest.approx(
            0.60 * 50.0 * 14.0 * 0.44 * (1.4 - 0.6), rel=1e-12
        )

    def test_the_J10_9_J10_10_branches_meet_at_0_4_Py(self) -> None:
        """(1.4 - 0.4) = 1.0 exactly -- the branches are continuous."""
        at_limit = J.panel_zone_shear(50.0, 14.0, 0.44, 400.0, 1000.0)
        just_above = J.panel_zone_shear(50.0, 14.0, 0.44, 400.0001, 1000.0)
        assert at_limit.nominal == pytest.approx(just_above.nominal, rel=1e-6)

    def test_flange_credit_needs_the_inelastic_model(self) -> None:
        """Eqs. J10-11/J10-12 credit the column flanges, but only when the
        analysis models inelastic panel-zone deformation."""
        without = J.panel_zone_shear(50.0, 14.0, 0.44, 100.0, 1000.0).nominal
        with_flanges = J.panel_zone_shear(
            50.0, 14.0, 0.44, 100.0, 1000.0, bcf=14.5, tcf=0.71, db=18.0,
            inelastic_deformation_considered=True,
        ).nominal
        assert with_flanges > without
        assert 1.15 < with_flanges / without < 1.5

    def test_flange_credit_requires_the_flange_dimensions(self) -> None:
        with pytest.raises(GeometryError, match="bcf, tcf and db"):
            J.panel_zone_shear(
                50.0, 14.0, 0.44, 100.0, 1000.0, inelastic_deformation_considered=True
            )

    def test_asd_alpha_raises_the_panel_zone_demand(self) -> None:
        lrfd = J.panel_zone_shear(50.0, 14.0, 0.44, 300.0, 1000.0, basis=Basis.LRFD)
        asd = J.panel_zone_shear(50.0, 14.0, 0.44, 300.0, 1000.0, basis=Basis.ASD)
        assert asd.governing.detail["alpha*Pr/Py"] == pytest.approx(0.48, rel=1e-12)
        assert lrfd.governing.detail["alpha*Pr/Py"] == pytest.approx(0.30, rel=1e-12)
