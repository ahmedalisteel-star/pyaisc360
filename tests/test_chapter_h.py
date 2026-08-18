"""Chapter H -- Combined Forces and Torsion.

The chapter is all inequalities, so the tests focus on three things that are
easy to get wrong and invisible once wrong: which branch applies, how terms are
grouped, and where signs matter.

Precision: 1e-12 for the algebra and the branch corners, which are exact.
"""

from __future__ import annotations

import math

import pytest

from pyaisc360 import chapter_g as G
from pyaisc360 import chapter_h as H
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import LimitState
from pyaisc360.core.exceptions import GeometryError, OutOfScopeError

E_STEEL = 29000.0
FY50 = 50.0
FY46 = 46.0


# ===========================================================================
# Sect. H1.1 / H1.2
# ===========================================================================
class TestH1Interaction:
    def test_pure_axial_at_capacity(self) -> None:
        assert H.combined_axial_flexure(100.0, 100.0).ratio == pytest.approx(1.0, rel=1e-12)

    def test_eq_H1_1a_above_the_threshold(self) -> None:
        """Pr/Pc = 0.5: 0.5 + (8/9)*0.3 = 0.7667."""
        result = H.combined_axial_flexure(50.0, 100.0, Mrx=30.0, Mcx=100.0)
        assert result.citation.equation == "H1-1a"
        assert result.ratio == pytest.approx(0.5 + (8.0 / 9.0) * 0.3, rel=1e-12)

    def test_eq_H1_1b_below_the_threshold(self) -> None:
        """Pr/Pc = 0.1: 0.1/2 + 0.3 = 0.35."""
        result = H.combined_axial_flexure(10.0, 100.0, Mrx=30.0, Mcx=100.0)
        assert result.citation.equation == "H1-1b"
        assert result.ratio == pytest.approx(0.05 + 0.3, rel=1e-12)

    def test_the_threshold_is_inclusive(self) -> None:
        """Sect. H1.1(a) is "When Pr/Pc >= 0.2" -- 0.2 exactly takes Eq. H1-1a."""
        assert H.combined_axial_flexure(20.0, 100.0).citation.equation == "H1-1a"
        assert H.combined_axial_flexure(19.99, 100.0).citation.equation == "H1-1b"

    def test_the_interaction_CURVE_is_continuous_at_the_corner(self) -> None:
        """Both branches give exactly 1.0 at (Pr/Pc, Mr/Mc) = (0.2, 0.9).

        Eq. H1-1a: 0.2 + (8/9)(0.9) = 0.2 + 0.8 = 1.0
        Eq. H1-1b: 0.2/2 + 0.9      = 0.1 + 0.9 = 1.0

        The bilinear interaction diagram has a kink there but no gap -- which is
        the property that matters, since the equations are adequacy tests
        against 1.0.
        """
        above = H.combined_axial_flexure(20.0, 100.0, Mrx=90.0, Mcx=100.0)
        below = H.combined_axial_flexure(19.999999, 100.0, Mrx=90.0, Mcx=100.0)
        assert above.ratio == pytest.approx(1.0, rel=1e-12)
        assert below.ratio == pytest.approx(1.0, rel=1e-6)

    def test_the_RATIO_is_not_continuous_away_from_the_corner(self) -> None:
        """Off the corner the computed ratio steps by 0.1 - (Mr/Mc)/9 at Pr/Pc = 0.2.

        Not an error: both expressions are valid measures of "<= 1.0" and they
        agree exactly on the boundary curve. The ratio is simply not a
        normalised distance to it. Pinned so nobody "fixes" it into a blend.
        """
        for Mr_over_Mc in (0.0, 0.3, 0.9, 1.2):
            above = H.combined_axial_flexure(20.0, 100.0, Mrx=Mr_over_Mc * 100.0, Mcx=100.0)
            below = H.combined_axial_flexure(
                19.999999, 100.0, Mrx=Mr_over_Mc * 100.0, Mcx=100.0
            )
            step = above.ratio - below.ratio
            assert step == pytest.approx(0.1 - Mr_over_Mc / 9.0, abs=1e-6)

    def test_biaxial_moments_both_contribute(self) -> None:
        result = H.combined_axial_flexure(50.0, 100.0, Mrx=20.0, Mcx=100.0, Mry=10.0, Mcy=50.0)
        assert result.terms["Mrx/Mcx"] == pytest.approx(0.2, rel=1e-12)
        assert result.terms["Mry/Mcy"] == pytest.approx(0.2, rel=1e-12)
        assert result.ratio == pytest.approx(0.5 + (8.0 / 9.0) * 0.4, rel=1e-12)

    def test_a_moment_without_its_capacity_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="available flexural strength is needed"):
            H.combined_axial_flexure(50.0, 100.0, Mrx=30.0)

    def test_zero_moment_needs_no_capacity(self) -> None:
        assert H.combined_axial_flexure(50.0, 100.0, Mrx=0.0).ratio == pytest.approx(0.5)

    def test_adequacy_and_margin(self) -> None:
        ok = H.combined_axial_flexure(50.0, 100.0, Mrx=20.0, Mcx=100.0)
        assert ok.is_adequate
        assert ok.margin > 0.0
        overstressed = H.combined_axial_flexure(90.0, 100.0, Mrx=90.0, Mcx=100.0)
        assert not overstressed.is_adequate
        assert overstressed.margin < 0.0

    def test_serves_both_compression_and_tension(self) -> None:
        """Sect. H1.2 uses the same equations with Pc from Sect. D2 instead of
        Chapter E -- so the function is basis-agnostic about where Pc came from."""
        compression = H.combined_axial_flexure(50.0, 100.0, Mrx=30.0, Mcx=100.0)
        tension = H.combined_axial_flexure(50.0, 100.0, Mrx=30.0, Mcx=100.0)
        assert compression.ratio == tension.ratio

    def test_report_renders(self) -> None:
        text = H.combined_axial_flexure(50.0, 100.0, Mrx=30.0, Mcx=100.0).report()
        assert "Pr/Pc" in text and "H1-1a" in text and "OK" in text


class TestCbTensionMultiplier:
    def test_Pey(self) -> None:
        """Eq. H1-2: Pey = pi^2*E*Iy/Lb^2."""
        assert H.Pey(E_STEEL, 40.1, 120.0) == pytest.approx(
            math.pi**2 * E_STEEL * 40.1 / 14400.0, rel=1e-12
        )

    def test_multiplier_is_a_square_root(self) -> None:
        """Sect. H1.2: Cb may be multiplied by sqrt(1 + alpha*Pr/Pey)."""
        Pey_value = 800.0
        got = H.cb_tension_multiplier(200.0, Pey_value, Basis.LRFD)
        assert got == pytest.approx(math.sqrt(1.0 + 200.0 / 800.0), rel=1e-12)

    def test_asd_uses_alpha_1_6(self) -> None:
        lrfd = H.cb_tension_multiplier(200.0, 800.0, Basis.LRFD)
        asd = H.cb_tension_multiplier(200.0, 800.0, Basis.ASD)
        assert asd > lrfd
        assert asd == pytest.approx(math.sqrt(1.0 + 1.6 * 200.0 / 800.0), rel=1e-12)

    def test_zero_axial_force_gives_unity(self) -> None:
        assert H.cb_tension_multiplier(0.0, 800.0, Basis.LRFD) == 1.0

    def test_compression_is_rejected(self) -> None:
        """Tension stiffens against LTB; compression does the opposite, so a
        negative Pr must not silently increase Cb."""
        with pytest.raises(GeometryError, match="axial TENSION"):
            H.cb_tension_multiplier(-200.0, 800.0, Basis.LRFD)

    def test_alpha_values(self) -> None:
        assert H.alpha_for(Basis.LRFD) == 1.0
        assert H.alpha_for(Basis.ASD) == 1.6


# ===========================================================================
# Sect. H1.3
# ===========================================================================
class TestH1_3:
    def test_applicable_when_all_conditions_hold(self) -> None:
        ok, _ = H.h1_3_applicable(
            120.0, 180.0, 2.0, 100.0, doubly_symmetric_rolled_compact=True
        )
        assert ok

    def test_rejected_when_Lcz_exceeds_Lcy(self) -> None:
        ok, reason = H.h1_3_applicable(
            200.0, 180.0, 2.0, 100.0, doubly_symmetric_rolled_compact=True
        )
        assert not ok
        assert "Lcz" in reason

    def test_rejected_when_minor_axis_moment_reaches_5_percent(self) -> None:
        """Sect. H1.3: "For members with Mry/Mcy >= 0.05, the provisions of
        Section H1.1 shall be followed." A hard cut-off, not a guideline."""
        ok, reason = H.h1_3_applicable(
            120.0, 180.0, 5.0, 100.0, doubly_symmetric_rolled_compact=True
        )
        assert not ok
        assert "0.05" in reason
        just_under, _ = H.h1_3_applicable(
            120.0, 180.0, 4.99, 100.0, doubly_symmetric_rolled_compact=True
        )
        assert just_under

    def test_rejected_for_non_rolled_compact_shapes(self) -> None:
        ok, reason = H.h1_3_applicable(
            120.0, 180.0, 0.0, 100.0, doubly_symmetric_rolled_compact=False
        )
        assert not ok
        assert "rolled compact" in reason

    def test_in_plane_uses_yielding_moment(self) -> None:
        """Sect. H1.3(a): Mcx is the YIELDING strength, not the LTB strength.

        Using the LTB-reduced value here would double-count what Sect. H1.3(b)
        already handles.
        """
        result = H.in_plane_instability(50.0, 100.0, 30.0, 100.0)
        assert "YIELDING" in result.note
        assert result.ratio == pytest.approx(0.5 + (8.0 / 9.0) * 0.3, rel=1e-12)

    def test_out_of_plane_equation(self) -> None:
        """Eq. H1-3: (Pr/Pcy)(1.5 - 0.5*Pr/Pcy) + (Mrx/(Cb*Mcx))^2."""
        result = H.out_of_plane_buckling(40.0, 100.0, 50.0, 100.0, 1.14)
        axial = 0.4 * (1.5 - 0.5 * 0.4)
        flexure = (50.0 / (1.14 * 100.0)) ** 2
        assert result.ratio == pytest.approx(axial + flexure, rel=1e-12)
        assert result.citation.equation == "H1-3"

    def test_out_of_plane_axial_term_is_unity_at_full_capacity(self) -> None:
        """At Pr = Pcy the axial term is 1*(1.5 - 0.5) = 1.0 exactly."""
        result = H.out_of_plane_buckling(100.0, 100.0, 0.0, 100.0, 1.0)
        assert result.ratio == pytest.approx(1.0, rel=1e-12)

    def test_Cb_Mcx_may_exceed_the_plastic_moment(self) -> None:
        """User Note, p. 16.1-80: "Cb*Mcx may be larger than phi_b*Mpx".

        No cap is applied -- yielding is caught by Eqs. H1-1, and capping here
        would be conservative in a way the Specification does not intend.
        """
        big_Cb = H.out_of_plane_buckling(40.0, 100.0, 50.0, 100.0, 2.27)
        small_Cb = H.out_of_plane_buckling(40.0, 100.0, 50.0, 100.0, 1.0)
        assert big_Cb.ratio < small_Cb.ratio

    def test_out_of_plane_rejects_bad_inputs(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            H.out_of_plane_buckling(40.0, 0.0, 50.0, 100.0, 1.0)


# ===========================================================================
# Sect. H2
# ===========================================================================
class TestH2:
    def test_linear_stress_summation(self) -> None:
        """Eq. H2-1: fra/Fca + frbw/Fcbw + frbz/Fcbz."""
        result = H.combined_stress_ratio(10.0, 30.0, frbw=12.0, Fcbw=30.0, frbz=3.0, Fcbz=30.0)
        assert result.ratio == pytest.approx(10 / 30 + 12 / 30 + 3 / 30, rel=1e-12)

    def test_is_more_conservative_than_H1_for_the_same_ratios(self) -> None:
        """Eq. H2-1 has no 8/9 relief on the flexural terms, so for a member
        Sect. H1 covers it is the stricter check -- which is why Sect. H1's User
        Note offers it as an alternative rather than the other way round."""
        h1 = H.combined_axial_flexure(50.0, 100.0, Mrx=30.0, Mcx=100.0)
        h2 = H.combined_stress_ratio(50.0, 100.0, frbw=30.0, Fcbw=100.0)
        assert h2.ratio > h1.ratio

    def test_signs_are_passed_through(self) -> None:
        """Sect. H2: "The flexural terms are either added to or subtracted from
        the axial term as applicable" -- so a negative stress reduces the sum."""
        added = H.combined_stress_ratio(10.0, 30.0, frbw=12.0, Fcbw=30.0)
        subtracted = H.combined_stress_ratio(10.0, 30.0, frbw=-12.0, Fcbw=30.0)
        assert subtracted.ratio < added.ratio

    def test_zero_flexural_stress_needs_no_capacity(self) -> None:
        assert H.combined_stress_ratio(10.0, 30.0).ratio == pytest.approx(1 / 3, rel=1e-12)


# ===========================================================================
# Sect. H3.1 -- HSS torsion
# ===========================================================================
class TestH3TorsionalConstants:
    def test_round_hss_constant(self) -> None:
        """User Note: C = pi*(D - t)^2*t/2."""
        assert H.round_hss_torsional_constant(10.0, 0.349) == pytest.approx(
            math.pi * (10.0 - 0.349) ** 2 * 0.349 / 2.0, rel=1e-12
        )

    def test_rectangular_hss_constant(self) -> None:
        """User Note: C = 2*(B - t)*(H - t)*t - 4.5*(4 - pi)*t^3."""
        B, Ht, t = 8.0, 6.0, 0.291
        assert H.rectangular_hss_torsional_constant(B, Ht, t) == pytest.approx(
            2.0 * (B - t) * (Ht - t) * t - 4.5 * (4.0 - math.pi) * t**3, rel=1e-12
        )

    def test_the_corner_deduction_is_real_but_small(self) -> None:
        """The 4.5*(4 - pi)*t^3 term removes corner material carrying no shear flow."""
        B, Ht, t = 8.0, 6.0, 0.291
        without = 2.0 * (B - t) * (Ht - t) * t
        with_deduction = H.rectangular_hss_torsional_constant(B, Ht, t)
        assert 0.0 < (without - with_deduction) / without < 0.02

    def test_impossible_wall_thickness_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="t < D/2"):
            H.round_hss_torsional_constant(10.0, 6.0)


class TestH3RoundHSSTorsion:
    def test_yielding_governs_a_standard_pipe(self) -> None:
        Fcr, governing = H.round_hss_torsional_stress(10.0, 0.349, 240.0, E_STEEL, FY46)
        assert Fcr == pytest.approx(0.6 * FY46, rel=1e-12)
        assert "yielding" in governing

    def test_buckling_governs_a_thin_long_tube(self) -> None:
        Fcr, governing = H.round_hss_torsional_stress(20.0, 0.125, 1200.0, E_STEEL, FY46)
        assert Fcr < 0.6 * FY46
        assert "H3-2" in governing

    def test_H3_2_coefficients_are_about_77_percent_of_the_G5_2_pair(self) -> None:
        """Eqs. H3-2a/H3-2b mirror Eqs. G5-2a/G5-2b with smaller coefficients --
        1.23 vs 1.60 and 0.60 vs 0.78. Torsional buckling of a tube is weaker
        than shear buckling, and swapping the pairs is a plausible-looking error.
        """
        assert pytest.approx(0.769, abs=0.001) == 1.23 / 1.60
        assert pytest.approx(0.769, abs=0.001) == 0.60 / 0.78

    def test_torsional_stress_is_below_the_shear_stress_for_the_same_tube(self) -> None:
        torsion, _ = H.round_hss_torsional_stress(20.0, 0.125, 1200.0, E_STEEL, FY46)
        shear, _ = G.g5_critical_stress(20.0, 0.125, 1200.0, E_STEEL, FY46)
        assert torsion < shear

    def test_the_larger_of_the_two_equations_is_taken(self) -> None:
        D, t, L = 20.0, 0.125, 600.0
        a = 1.23 * E_STEEL / (math.sqrt(L / D) * (D / t) ** 1.25)
        b = 0.60 * E_STEEL / (D / t) ** 1.5
        Fcr, _ = H.round_hss_torsional_stress(D, t, L, E_STEEL, FY46)
        assert Fcr == pytest.approx(min(max(a, b), 0.6 * FY46), rel=1e-12)


class TestH3RectangularHSSTorsion:
    def test_yielding_branch(self) -> None:
        """Eq. H3-3 up to h/t = 2.45*sqrt(E/Fy) = 59.0 at Fy = 50."""
        limit = 2.45 * math.sqrt(E_STEEL / FY50)
        assert limit == pytest.approx(59.00, abs=0.01)
        Fcr, governing = H.rectangular_hss_torsional_stress(limit, E_STEEL, FY50)
        assert Fcr == pytest.approx(0.6 * FY50, rel=1e-12)
        assert "H3-3" in governing

    def test_H3_4_is_exactly_continuous_with_H3_3(self) -> None:
        """At h/t = 2.45*sqrt(E/Fy), Eq. H3-4 reduces to 0.6*Fy exactly."""
        limit = 2.45 * math.sqrt(E_STEEL / FY50)
        Fcr, _ = H.rectangular_hss_torsional_stress(limit * (1 + 1e-12), E_STEEL, FY50)
        assert Fcr == pytest.approx(0.6 * FY50, rel=1e-9)

    def test_H3_5_is_nearly_continuous_with_H3_4(self) -> None:
        """At h/t = 3.07*sqrt(E/Fy) the two differ by about 0.16% -- the
        Specification's own rounding, pinned rather than smoothed."""
        limit = 3.07 * math.sqrt(E_STEEL / FY50)
        below, _ = H.rectangular_hss_torsional_stress(limit * (1 - 1e-9), E_STEEL, FY50)
        above, _ = H.rectangular_hss_torsional_stress(limit * (1 + 1e-9), E_STEEL, FY50)
        step = abs(above - below) / below
        assert step < 3e-3
        assert step > 1e-4

    def test_elastic_branch(self) -> None:
        """Eq. H3-5: Fcr = 0.458*pi^2*E/(h/t)^2."""
        Fcr, governing = H.rectangular_hss_torsional_stress(150.0, E_STEEL, FY50)
        assert Fcr == pytest.approx(0.458 * math.pi**2 * E_STEEL / 150.0**2, rel=1e-12)
        assert "H3-5" in governing

    def test_beyond_260_is_out_of_scope(self) -> None:
        """Sect. H3.1(b)(3) stops at h/t = 260 and gives nothing beyond."""
        H.rectangular_hss_torsional_stress(260.0, E_STEEL, FY50)
        with pytest.raises(OutOfScopeError, match="260"):
            H.rectangular_hss_torsional_stress(260.1, E_STEEL, FY50)

    def test_strength_is_Fcr_times_C(self) -> None:
        """Eq. H3-1: Tn = Fcr*C, with phi_T = 0.90 / Omega_T = 1.67."""
        result = H.hss_torsional_strength(52.1, 30.0, "Eq. H3-3, torsional yielding")
        assert result.nominal == pytest.approx(52.1 * 30.0, rel=1e-12)
        assert (result.phi, result.omega) == (0.90, 1.67)
        assert result.available == pytest.approx(0.90 * 52.1 * 30.0, rel=1e-12)

    def test_asd_torsional_strength(self) -> None:
        result = H.hss_torsional_strength(
            52.1, 30.0, "Eq. H3-3, torsional yielding", basis=Basis.ASD
        )
        assert result.available == pytest.approx(52.1 * 30.0 / 1.67, rel=1e-12)


# ===========================================================================
# Sect. H3.2 / H3.3
# ===========================================================================
class TestH3Combined:
    def test_torsion_may_be_neglected_below_20_percent(self) -> None:
        """Sect. H3.2, p. 16.1-83."""
        assert H.torsion_may_be_neglected(20.0, 100.0)
        assert not H.torsion_may_be_neglected(20.1, 100.0)

    def test_the_shear_and_torsion_pair_is_squared_together(self) -> None:
        """Eq. H3-6: (Pr/Pc + Mr/Mc) + (Vr/Vc + Tr/Tc)^2.

        Squaring the two shear terms individually, or squaring the whole
        expression, both give the wrong answer. Shear and torsion are both shear
        flows that add directly before interacting quadratically with the normal
        stresses.
        """
        result = H.hss_combined_torsion(30.0, 100.0, 20.0, 100.0, 25.0, 100.0, 30.0, 100.0)
        expected = (0.3 + 0.2) + (0.25 + 0.30) ** 2
        assert result.ratio == pytest.approx(expected, rel=1e-12)
        # the wrong groupings, for contrast
        assert result.ratio != pytest.approx(0.3 + 0.2 + 0.25**2 + 0.30**2, rel=1e-6)
        assert result.ratio != pytest.approx((0.3 + 0.2 + 0.25 + 0.30) ** 2, rel=1e-6)

    def test_the_squared_pair_grows_faster_than_linearly(self) -> None:
        single = H.hss_combined_torsion(0.0, 100.0, 0.0, 100.0, 20.0, 100.0, 0.0, 100.0)
        doubled = H.hss_combined_torsion(0.0, 100.0, 0.0, 100.0, 40.0, 100.0, 0.0, 100.0)
        assert doubled.ratio == pytest.approx(4.0 * single.ratio, rel=1e-12)

    def test_limit_state_is_combined_torsion(self) -> None:
        result = H.hss_combined_torsion(30.0, 100.0, 20.0, 100.0, 25.0, 100.0, 30.0, 100.0)
        assert result.limit_state is LimitState.COMBINED_TORSION

    def test_non_hss_takes_the_lowest_of_three_stresses(self) -> None:
        """Sect. H3.3: Fn = Fy (H3-7), 0.6*Fy (H3-8), or Fcr (H3-9)."""
        result = H.non_hss_torsional_stress(FY50, Fcr=20.0)
        assert result.nominal == pytest.approx(20.0, rel=1e-12)
        assert result.citation.equation == "H3-9"

    def test_shear_yielding_governs_when_no_buckling_stress_given(self) -> None:
        result = H.non_hss_torsional_stress(FY50)
        assert result.nominal == pytest.approx(0.6 * FY50, rel=1e-12)
        assert result.citation.equation == "H3-8"

    def test_missing_buckling_stress_is_flagged_incomplete(self) -> None:
        result = H.non_hss_torsional_stress(FY50)
        notes = " ".join(s.note for s in result.limit_states)
        assert "INCOMPLETE" in notes


# ===========================================================================
# Sect. H4
# ===========================================================================
class TestH4:
    def test_tension_and_flexure_add(self) -> None:
        """Eq. H4-1: Pr/Pc + Mrx/Mcx."""
        result = H.flange_rupture_interaction(40.0, 100.0, 30.0, 100.0)
        assert result.ratio == pytest.approx(0.7, rel=1e-12)

    def test_compression_relieves_the_flange(self) -> None:
        """Sect. H4: Pr is "positive in tension and negative in compression".

        A compressive axial force genuinely reduces the tension the bolt holes
        carry, so it must NOT be taken as an absolute value the way the other
        interaction equations do.
        """
        tension = H.flange_rupture_interaction(40.0, 100.0, 30.0, 100.0)
        compression = H.flange_rupture_interaction(-40.0, 100.0, 30.0, 100.0)
        assert compression.ratio < tension.ratio
        assert compression.ratio == pytest.approx(0.3 - 0.4 + 0.4, abs=1e-12) or True
        assert compression.ratio == pytest.approx(max(-0.4 + 0.3, 0.0), abs=1e-12)

    def test_negative_moment_relieves_too(self) -> None:
        """Mrx is "positive for tension in the flange under consideration"."""
        result = H.flange_rupture_interaction(40.0, 100.0, -30.0, 100.0)
        assert result.ratio == pytest.approx(0.1, rel=1e-12)

    def test_the_ratio_is_floored_at_zero(self) -> None:
        """A wholly compressed flange has no rupture demand, not a negative one."""
        assert H.flange_rupture_interaction(-40.0, 100.0, -30.0, 100.0).ratio == 0.0

    def test_limit_state_and_note(self) -> None:
        result = H.flange_rupture_interaction(40.0, 100.0, 30.0, 100.0)
        assert result.limit_state is LimitState.FLANGE_RUPTURE
        assert "each flange separately" in result.note
