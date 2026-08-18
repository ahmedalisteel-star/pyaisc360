"""Appendices 1-4 and 6-8 -- the whole appendix set.

Appendices 6, 7 and 8 are tested together because they interlock: Appendix 7
gates its two methods on a drift ratio that its own User Note says may be taken
as Appendix 8's ``B2``, and both Appendix 7 methods lean on Appendix 8's ``B1``.
Appendices 1 through 4 -- inelastic analysis, ponding, fatigue and fire -- are
independent of each other and of those three.

Precision: 1e-12 for closed forms; 0.5% where a value traces to a rounded
published figure, and exact equality for the printed table constants.
"""

from __future__ import annotations

import math

import pytest

from pyaisc360 import appendix_1 as A1
from pyaisc360 import appendix_2 as A2
from pyaisc360 import appendix_3 as A3
from pyaisc360 import appendix_4 as A4
from pyaisc360 import appendix_6 as A6
from pyaisc360 import appendix_7 as A7
from pyaisc360 import appendix_8 as A8
from pyaisc360.core.config import Basis
from pyaisc360.core.exceptions import AISC360Error, GeometryError, OutOfScopeError

E_STEEL = 29000.0


# ===========================================================================
# Appendix 6 -- Member Stability Bracing
# ===========================================================================
class TestAppendix6Factors:
    def test_phi_is_0_75_not_0_90(self) -> None:
        """Bracing uses phi = 0.75, not the member chapters' 0.90."""
        assert A6.PHI_BRACING == 0.75
        assert A6.OMEGA_BRACING == 2.00

    def test_torsional_omega_is_3_not_2(self) -> None:
        """Eq. A-6-11 User Note, p. 16.1-242: Omega = 1.5^2/phi = 3.00, "because
        the moment term is squared". Using 2.00 understates it by a third."""
        assert A6.OMEGA_TORSIONAL == 3.00
        assert pytest.approx(1.5**2 / A6.PHI_BRACING, rel=1e-12) == A6.OMEGA_TORSIONAL

    def test_lrfd_stiffness_factor_is_one_over_phi(self) -> None:
        req = A6.column_point_bracing(100.0, 120.0, Basis.LRFD)
        assert req.stiffness == pytest.approx((1.0 / 0.75) * 8.0 * 100.0 / 120.0, rel=1e-12)

    def test_asd_stiffness_factor_is_omega(self) -> None:
        req = A6.column_point_bracing(100.0, 120.0, Basis.ASD)
        assert req.stiffness == pytest.approx(2.00 * 8.0 * 100.0 / 120.0, rel=1e-12)


class TestColumnBracing:
    def test_panel_strength_and_stiffness(self) -> None:
        """Eqs. A-6-1 and A-6-2a: Vbr = 0.005*Pr, beta = (1/phi)(2Pr/Lbr)."""
        req = A6.column_panel_bracing(200.0, 144.0)
        assert req.strength == pytest.approx(0.005 * 200.0, rel=1e-12)
        assert req.stiffness == pytest.approx((1 / 0.75) * 2 * 200.0 / 144.0, rel=1e-12)
        assert req.strength_citation.equation == "A-6-1"
        assert req.stiffness_citation.equation == "A-6-2a"

    def test_point_strength_and_stiffness(self) -> None:
        """Eqs. A-6-3 and A-6-4a: Pbr = 0.01*Pr, beta = (1/phi)(8Pr/Lbr)."""
        req = A6.column_point_bracing(200.0, 144.0)
        assert req.strength == pytest.approx(0.01 * 200.0, rel=1e-12)
        assert req.stiffness == pytest.approx((1 / 0.75) * 8 * 200.0 / 144.0, rel=1e-12)

    def test_point_needs_twice_the_strength_and_four_times_the_stiffness(self) -> None:
        """A point brace works alone; a panel brace shares a whole segment's
        deviation. The 2x / 4x ratio is the price of that isolation."""
        panel = A6.column_panel_bracing(200.0, 144.0)
        point = A6.column_point_bracing(200.0, 144.0)
        assert point.strength / panel.strength == pytest.approx(2.0, rel=1e-12)
        assert point.stiffness / panel.stiffness == pytest.approx(4.0, rel=1e-12)

    def test_asd_uses_the_b_equations(self) -> None:
        assert A6.column_panel_bracing(200.0, 144.0, Basis.ASD).stiffness_citation.equation == (
            "A-6-2b"
        )
        assert A6.column_point_bracing(200.0, 144.0, Basis.ASD).stiffness_citation.equation == (
            "A-6-4b"
        )

    def test_stiffness_rises_as_bracing_gets_closer(self) -> None:
        """beta goes as 1/Lbr -- closely spaced braces each need MORE stiffness,
        which is why Sect. 6.2.2 lets Lbr be the maximum permitted Lc instead."""
        close = A6.column_point_bracing(200.0, 60.0)
        far = A6.column_point_bracing(200.0, 240.0)
        assert close.stiffness == pytest.approx(4.0 * far.stiffness, rel=1e-12)
        assert close.strength == far.strength

    def test_panel_note_flags_the_connection_requirement(self) -> None:
        assert "point-brace force" in A6.column_panel_bracing(200.0, 144.0).note

    def test_rejects_bad_geometry(self) -> None:
        with pytest.raises(GeometryError, match="unbraced length"):
            A6.column_point_bracing(200.0, 0.0)


class TestBeamLateralBracing:
    def test_panel_uses_the_flange_force(self) -> None:
        """Eq. A-6-5: Vbr = 0.01*(Mr*Cd/ho). Mr/ho is the flange force."""
        req = A6.beam_panel_bracing(3000.0, 17.4, 120.0)
        assert req.strength == pytest.approx(0.01 * 3000.0 / 17.4, rel=1e-12)

    def test_point_needs_twice_the_strength_and_2_5_times_the_stiffness(self) -> None:
        """Eqs. A-6-7/A-6-8 against A-6-5/A-6-6: 0.02 vs 0.01, and 10 vs 4."""
        panel = A6.beam_panel_bracing(3000.0, 17.4, 120.0)
        point = A6.beam_point_bracing(3000.0, 17.4, 120.0)
        assert point.strength / panel.strength == pytest.approx(2.0, rel=1e-12)
        assert point.stiffness / panel.stiffness == pytest.approx(2.5, rel=1e-12)

    def test_Cd_doubles_the_requirement_near_an_inflection_point(self) -> None:
        """Sect. 6.3.1a: Cd = 2.0 for the brace closest to the inflection point
        in double curvature -- that brace restrains a flange changing from
        compression to tension across it."""
        plain = A6.beam_point_bracing(3000.0, 17.4, 120.0, Cd=A6.CD_DEFAULT)
        inflection = A6.beam_point_bracing(3000.0, 17.4, 120.0, Cd=A6.CD_NEAR_INFLECTION)
        assert inflection.strength == pytest.approx(2.0 * plain.strength, rel=1e-12)
        assert inflection.stiffness == pytest.approx(2.0 * plain.stiffness, rel=1e-12)

    def test_only_the_two_published_Cd_values_are_accepted(self) -> None:
        with pytest.raises(AISC360Error, match="Cd is 1.0, or 2.0"):
            A6.beam_point_bracing(3000.0, 17.4, 120.0, Cd=1.5)

    def test_ho_is_the_flange_centroid_distance_not_the_depth(self) -> None:
        """Using d in place of ho understates the flange force by a few percent."""
        by_ho = A6.beam_point_bracing(3000.0, 17.4, 120.0)
        by_depth = A6.beam_point_bracing(3000.0, 18.0, 120.0)
        assert by_ho.strength > by_depth.strength


class TestBeamTorsionalBracing:
    def test_strength_is_two_percent_of_the_moment(self) -> None:
        """Eq. A-6-9: Mbr = 0.02*Mr -- a moment, not a force."""
        req = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None)
        assert req.strength == pytest.approx(0.02 * 3000.0, rel=1e-12)
        assert req.kind == "torsional"

    def test_infinite_beta_sec_gives_beta_br_equals_beta_T(self) -> None:
        """Sect. 6.3.2a: beta_sec may be taken as infinity when a cross-frame is
        attached near both flanges."""
        req = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None)
        beta_T = (1 / 0.75) * (2.4 * 360.0 / (3 * E_STEEL * 40.1)) * 3000.0**2
        assert req.stiffness == pytest.approx(beta_T, rel=1e-12)
        assert "infinity" in req.note

    def test_finite_beta_sec_increases_the_requirement(self) -> None:
        """Eq. A-6-10: beta_br = beta_T/(1 - beta_T/beta_sec) > beta_T always."""
        rigid = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None)
        flexible = A6.beam_torsional_bracing(
            3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, rigid.stiffness * 4.0
        )
        assert flexible.stiffness > rigid.stiffness

    def test_inadequate_web_stiffness_raises(self) -> None:
        """User Note, p. 16.1-243: when beta_sec < beta_T, Eq. A-6-10 goes
        negative, meaning "torsional beam bracing will not be effective due to
        inadequate web distortional stiffness". No brace stiffness fixes it."""
        rigid = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None)
        with pytest.raises(OutOfScopeError, match="web distortional stiffness"):
            A6.beam_torsional_bracing(
                3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, rigid.stiffness * 0.5
            )

    def test_Mr_over_Cb_is_squared_after_dividing(self) -> None:
        """Eq. A-6-11: (Mr/Cb)^2 -- Cb divides BEFORE squaring, so it is a
        quadratic relief, not a linear one."""
        cb_one = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None)
        cb_two = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 2.0, None)
        assert cb_one.stiffness / cb_two.stiffness == pytest.approx(4.0, rel=1e-12)

    def test_asd_uses_omega_3(self) -> None:
        lrfd = A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None)
        asd = A6.beam_torsional_bracing(
            3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None, basis=Basis.ASD
        )
        assert asd.stiffness / lrfd.stiffness == pytest.approx(3.00 * 0.75, rel=1e-12)

    def test_web_distortional_stiffness(self) -> None:
        """Eq. A-6-12: beta_sec = (3.3E/ho)(1.5*ho*tw^3/12 + tst*bs^3/12)."""
        got = A6.web_distortional_stiffness(E_STEEL, 17.4, 0.355, tst=0.25, bs=3.0)
        expected = (3.3 * E_STEEL / 17.4) * (
            1.5 * 17.4 * 0.355**3 / 12.0 + 0.25 * 3.0**3 / 12.0
        )
        assert got == pytest.approx(expected, rel=1e-12)

    def test_a_stiffener_raises_beta_sec_substantially(self) -> None:
        bare = A6.web_distortional_stiffness(E_STEEL, 17.4, 0.355)
        stiffened = A6.web_distortional_stiffness(E_STEEL, 17.4, 0.355, tst=0.25, bs=3.0)
        assert stiffened > 3.0 * bare

    def test_continuous_form_drops_the_stiffener_term(self) -> None:
        """Eq. A-6-13: beta_sec = 3.3*E*tw^3/(12*ho)."""
        assert A6.continuous_web_distortional_stiffness(
            E_STEEL, 17.4, 0.355
        ) == pytest.approx(3.3 * E_STEEL * 0.355**3 / (12.0 * 17.4), rel=1e-12)


class TestBeamColumnBracing:
    def test_requirements_sum(self) -> None:
        """Sect. 6.4(a)/(b): strengths add and stiffnesses add."""
        axial = A6.column_point_bracing(200.0, 144.0)
        flexural = A6.beam_point_bracing(3000.0, 17.4, 144.0)
        combined = A6.beam_column_bracing(axial, flexural)
        assert combined.strength == pytest.approx(axial.strength + flexural.strength, rel=1e-12)
        assert combined.stiffness == pytest.approx(
            axial.stiffness + flexural.stiffness, rel=1e-12
        )

    def test_mixing_panel_and_point_is_rejected(self) -> None:
        with pytest.raises(AISC360Error, match="Sect. 6.4"):
            A6.beam_column_bracing(
                A6.column_panel_bracing(200.0, 144.0),
                A6.beam_point_bracing(3000.0, 17.4, 144.0),
            )

    def test_torsional_plus_axial_is_rejected_as_case_c(self) -> None:
        """Sect. 6.4(c) is engineering judgement, not a sum."""
        with pytest.raises(AISC360Error, match="6.4\\(c\\)"):
            A6.beam_column_bracing(
                A6.column_point_bracing(200.0, 144.0),
                A6.beam_torsional_bracing(3000.0, 360.0, 3, E_STEEL, 40.1, 1.0, None),
            )

    def test_note_records_the_withdrawn_relaxation(self) -> None:
        """Sect. 6.4(b): for beam-columns, Lbr is the ACTUAL unbraced length --
        the Sects. 6.2.2 and 6.3.1b relaxations do not apply."""
        combined = A6.beam_column_bracing(
            A6.column_point_bracing(200.0, 144.0), A6.beam_point_bracing(3000.0, 17.4, 144.0)
        )
        assert "ACTUAL unbraced length" in combined.note

    def test_report_renders_both_requirements(self) -> None:
        text = A6.column_point_bracing(200.0, 144.0).report()
        assert "required strength" in text and "required stiffness" in text
        assert "A-6-3" in text and "A-6-4a" in text


# ===========================================================================
# Appendix 7 -- Alternative Methods of Design for Stability
# ===========================================================================
class TestAppendix7:
    def test_the_two_drift_thresholds_are_different(self) -> None:
        """1.5 gates whether the method may be used at all (Sects. 7.2.1(b),
        7.3.1(b)); 1.1 gates whether K = 1.0 may be used without a sidesway
        buckling analysis (Sect. 7.2.3(b) Exception)."""
        assert A7.DRIFT_RATIO_LIMIT == 1.5
        assert A7.K_EQUAL_ONE_DRIFT_LIMIT == 1.1

    def test_effective_length_method_permitted_below_1_5(self) -> None:
        assert A7.effective_length_method_permitted(1.5).permitted
        assert not A7.effective_length_method_permitted(1.51).permitted

    def test_failure_names_the_condition(self) -> None:
        check = A7.effective_length_method_permitted(2.0)
        assert "7.2.1(b)" in check.failures[0]
        with pytest.raises(OutOfScopeError, match="not permitted"):
            check.require()

    def test_nominal_stiffness_is_recorded_in_the_note(self) -> None:
        """Sect. 7.2.2: the Sect. C2.1(a) stiffness reduction is NOT applied."""
        assert "NOMINAL stiffness" in A7.effective_length_method_permitted(1.2).note

    def test_K_is_unity_for_braced_frames_at_any_drift(self) -> None:
        """Sect. 7.2.3(a): braced frames and shear walls always take K = 1.0."""
        assert A7.K_may_be_taken_as_unity(3.0, braced_frame=True)

    def test_K_is_unity_for_moment_frames_only_below_1_1(self) -> None:
        assert A7.K_may_be_taken_as_unity(1.1)
        assert not A7.K_may_be_taken_as_unity(1.11)

    def test_axial_limit_is_half_the_cross_section_strength(self) -> None:
        """Eq. A-7-1: alpha*Pr <= 0.5*Pns, where Pns = Fy*Ag (nonslender) --
        the CROSS-SECTION strength, with no buckling reduction."""
        assert A7.axial_force_limit(1000.0) == pytest.approx(500.0, rel=1e-12)

    def test_first_order_method_gated_on_axial_force(self) -> None:
        assert A7.first_order_method_permitted(1.2, 400.0, 1000.0).permitted
        failed = A7.first_order_method_permitted(1.2, 600.0, 1000.0)
        assert not failed.permitted
        assert "A-7-1" in failed.failures[0]

    def test_asd_alpha_tightens_the_axial_limit(self) -> None:
        """alpha = 1.6 under ASD, so a service-level 400 kips is 640 effective."""
        assert A7.first_order_method_permitted(1.2, 400.0, 1000.0, Basis.LRFD).permitted
        assert not A7.first_order_method_permitted(1.2, 400.0, 1000.0, Basis.ASD).permitted

    def test_notional_load_drift_term(self) -> None:
        """Eq. A-7-2: Ni = 2.1*alpha*(Delta/L)*Yi."""
        Ni, governing = A7.additional_lateral_load(1000.0, 0.01)
        assert Ni == pytest.approx(2.1 * 0.01 * 1000.0, rel=1e-12)
        assert "drift-based" in governing

    def test_notional_load_floor(self) -> None:
        """The 0.0042*Yi floor is the same notional load Sect. C2.2b applies."""
        Ni, governing = A7.additional_lateral_load(1000.0, 0.001)
        assert Ni == pytest.approx(0.0042 * 1000.0, rel=1e-12)
        assert "floor" in governing

    def test_the_floor_binds_below_a_drift_ratio_of_0_002(self) -> None:
        """2.1*(Delta/L) = 0.0042 at Delta/L = 0.002 exactly."""
        at_crossover, _ = A7.additional_lateral_load(1000.0, 0.002)
        assert at_crossover == pytest.approx(0.0042 * 1000.0, rel=1e-12)

    def test_asd_alpha_raises_the_notional_load(self) -> None:
        lrfd, _ = A7.additional_lateral_load(1000.0, 0.01, Basis.LRFD)
        asd, _ = A7.additional_lateral_load(1000.0, 0.01, Basis.ASD)
        assert asd == pytest.approx(1.6 * lrfd, rel=1e-12)

    def test_first_order_note_records_the_B1_requirement(self) -> None:
        """Sect. 7.3.2(b): B1 must still be applied to the TOTAL member moments."""
        assert "B1" in A7.first_order_method_permitted(1.2, 400.0, 1000.0).note


# ===========================================================================
# Appendix 8 -- Approximate Second-Order Analysis
# ===========================================================================
class TestCmFactor:
    def test_single_curvature_gives_unity(self) -> None:
        """M1/M2 = -1 (single curvature): Cm = 0.6 - 0.4(-1) = 1.0.

        The peak moment sits near midspan where the P-delta deflection is
        largest, so no relief applies.
        """
        assert A8.Cm_no_transverse_load(-1.0) == pytest.approx(1.0, rel=1e-12)

    def test_reverse_curvature_gives_0_2(self) -> None:
        """M1/M2 = +1 (reverse curvature): Cm = 0.6 - 0.4 = 0.2.

        The peaks are at the ends where the deflection is zero, so the
        amplification nearly vanishes.
        """
        assert A8.Cm_no_transverse_load(1.0) == pytest.approx(0.2, rel=1e-12)

    def test_one_end_pinned(self) -> None:
        assert A8.Cm_no_transverse_load(0.0) == pytest.approx(0.6, rel=1e-12)

    def test_the_sign_convention_matches_Eq_F13_8(self) -> None:
        """Both Eq. A-8-4 and Eq. F13-8 take M1/M2 positive for REVERSE
        curvature -- so a sign can be shared between them without conversion."""
        from pyaisc360 import chapter_f as F

        reverse_Cm = A8.Cm_no_transverse_load(1.0)
        single_Cm = A8.Cm_no_transverse_load(-1.0)
        reverse_Lm = F.Lm_moment_redistribution(1.0, 1.65, E_STEEL, 50.0)
        single_Lm = F.Lm_moment_redistribution(-1.0, 1.65, E_STEEL, 50.0)
        # Reverse curvature is the benign case in both: lower Cm, longer Lm.
        assert reverse_Cm < single_Cm
        assert reverse_Lm > single_Lm

    def test_ratio_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="SMALLER end moment"):
            A8.Cm_no_transverse_load(1.5)


class TestB1Multiplier:
    def test_Pe1(self) -> None:
        """Eq. A-8-5: Pe1 = pi^2*EI*/Lc1^2."""
        EI = E_STEEL * 800.0
        assert A8.Pe1(EI, 180.0) == pytest.approx(math.pi**2 * EI / 180.0**2, rel=1e-12)

    def test_B1_amplifies(self) -> None:
        """Eq. A-8-3: B1 = Cm/(1 - alpha*Pr/Pe1)."""
        got = A8.B1_multiplier(1.0, 200.0, 1000.0, Basis.LRFD)
        assert got == pytest.approx(1.0 / (1.0 - 0.2), rel=1e-12)

    def test_B1_is_floored_at_unity(self) -> None:
        """A low Cm from strong reverse curvature would otherwise return an
        amplifier below 1 and REDUCE the design moment."""
        got = A8.B1_multiplier(0.2, 100.0, 1000.0, Basis.LRFD)
        assert 0.2 / (1.0 - 0.1) < 1.0
        assert got == 1.0

    def test_tension_members_take_B1_of_unity(self) -> None:
        """Sect. 8.2: "B1 shall be taken as 1.0 for members not subject to
        compression"."""
        assert A8.B1_multiplier(1.0, -200.0, 1000.0, Basis.LRFD) == 1.0
        assert A8.B1_multiplier(1.0, 0.0, 1000.0, Basis.LRFD) == 1.0

    def test_asd_alpha_increases_the_amplifier(self) -> None:
        lrfd = A8.B1_multiplier(1.0, 200.0, 1000.0, Basis.LRFD)
        asd = A8.B1_multiplier(1.0, 200.0, 1000.0, Basis.ASD)
        assert asd == pytest.approx(1.0 / (1.0 - 1.6 * 0.2), rel=1e-12)
        assert asd > lrfd

    def test_reaching_the_buckling_load_raises(self) -> None:
        with pytest.raises(OutOfScopeError, match="elastic buckling load"):
            A8.B1_multiplier(1.0, 1000.0, 1000.0, Basis.LRFD)

    def test_reduced_stiffness_matters(self) -> None:
        """EI* is 0.8*tau_b*EI under the direct analysis method. Passing the
        unreduced EI overstates Pe1 and understates B1."""
        EI = E_STEEL * 800.0
        full = A8.B1_multiplier(1.0, 200.0, A8.Pe1(EI, 180.0), Basis.LRFD)
        reduced = A8.B1_multiplier(1.0, 200.0, A8.Pe1(0.8 * EI, 180.0), Basis.LRFD)
        assert reduced > full


class TestB2Multiplier:
    def test_RM_is_unity_for_a_braced_frame(self) -> None:
        """Eq. A-8-8 with Pmf = 0: RM = 1.0."""
        assert A8.RM_factor(0.0, 1000.0) == pytest.approx(A8.RM_BRACED, rel=1e-12)

    def test_RM_lower_bound_is_0_85(self) -> None:
        """The User Note gives 0.85 as the lower bound -- reached at Pmf = Pstory."""
        assert A8.RM_factor(1000.0, 1000.0) == pytest.approx(
            A8.RM_MOMENT_FRAME_LOWER_BOUND, rel=1e-12
        )

    def test_RM_rejects_Pmf_above_Pstory(self) -> None:
        with pytest.raises(GeometryError, match="Pmf <= Pstory"):
            A8.RM_factor(1200.0, 1000.0)

    def test_story_buckling_strength(self) -> None:
        """Eq. A-8-7: Pe,story = RM*H*L/delta_H -- a story stiffness times a height."""
        assert A8.story_buckling_strength(0.85, 100.0, 144.0, 0.5) == pytest.approx(
            0.85 * 100.0 * 144.0 / 0.5, rel=1e-12
        )

    def test_B2_amplifies(self) -> None:
        """Eq. A-8-6: B2 = 1/(1 - alpha*Pstory/Pe,story)."""
        assert A8.B2_multiplier(2000.0, 20000.0, Basis.LRFD) == pytest.approx(
            1.0 / (1.0 - 0.1), rel=1e-12
        )

    def test_B2_is_floored_at_unity(self) -> None:
        assert A8.B2_multiplier(0.0, 20000.0, Basis.LRFD) == 1.0

    def test_sidesway_buckling_raises(self) -> None:
        with pytest.raises(OutOfScopeError, match="sidesway buckling"):
            A8.B2_multiplier(20000.0, 20000.0, Basis.LRFD)

    def test_B2_is_the_drift_ratio_Appendix_7_tests(self) -> None:
        """Appendix 7's User Note: the second-order/first-order drift ratio "may
        be taken as the B2 multiplier". A frame at B2 = 1.5 sits exactly on the
        limit for both alternative methods."""
        Pe_story = 20000.0
        Pstory = Pe_story / 3.0  # B2 = 1/(1 - 1/3) = 1.5
        B2 = A8.B2_multiplier(Pstory, Pe_story, Basis.LRFD)
        assert pytest.approx(1.5, rel=1e-12) == B2
        assert A7.effective_length_method_permitted(B2).permitted
        assert not A7.effective_length_method_permitted(B2 * 1.001).permitted


class TestAmplifiedForces:
    def test_the_amplifiers_apply_to_different_terms(self) -> None:
        """Eqs. A-8-1 and A-8-2: Mr = B1*Mnt + B2*Mlt, Pr = Pnt + B2*Plt.

        Note the asymmetry -- Pnt is NOT amplified. B1 is a moment effect: a
        member's own curvature does not change the axial force it carries.
        """
        got = A8.amplified_forces(Mnt=1000.0, Mlt=400.0, Pnt=300.0, Plt=80.0, B1=1.2, B2=1.4)
        assert got.Mr == pytest.approx(1.2 * 1000.0 + 1.4 * 400.0, rel=1e-12)
        assert got.Pr == pytest.approx(300.0 + 1.4 * 80.0, rel=1e-12)

    def test_B1_does_not_touch_the_translation_moment(self) -> None:
        """Applying B1 to Mlt would double-count with B2."""
        a = A8.amplified_forces(1000.0, 400.0, 300.0, 80.0, B1=1.2, B2=1.4)
        b = A8.amplified_forces(1000.0, 400.0, 300.0, 80.0, B1=2.0, B2=1.4)
        assert b.Mr - a.Mr == pytest.approx(0.8 * 1000.0, rel=1e-12)

    def test_a_braced_frame_has_no_translation_terms(self) -> None:
        got = A8.amplified_forces(1000.0, 0.0, 300.0, 0.0, B1=1.2, B2=1.0)
        assert got.Mr == pytest.approx(1200.0, rel=1e-12)
        assert got.Pr == pytest.approx(300.0, rel=1e-12)

    def test_amplifiers_below_unity_are_rejected(self) -> None:
        with pytest.raises(AISC360Error, match="cannot fall below 1.0"):
            A8.amplified_forces(1000.0, 400.0, 300.0, 80.0, B1=0.9, B2=1.4)

    def test_report_renders(self) -> None:
        text = A8.amplified_forces(1000.0, 400.0, 300.0, 80.0, B1=1.2, B2=1.4).report()
        assert "A-8-1" in text and "A-8-2" in text


# ===========================================================================
# Appendix 1 -- Design by Advanced Analysis
# ===========================================================================
class TestAppendix1:
    FY = 50.0

    def test_zero_axial_load_recovers_Table_B4_1b_case_15(self) -> None:
        """Eq. A-1-1 at Pu = 0 gives 3.76*sqrt(E/Fy) -- exactly lambda_p for a
        doubly symmetric I-shape web. Axial load then tightens it."""
        limit, note = A1.lambda_pd_web(0.0, 1000.0, E_STEEL, self.FY)
        assert limit == pytest.approx(3.76 * math.sqrt(E_STEEL / self.FY), rel=1e-12)
        assert "A-1-1" in note

    def test_axial_load_tightens_the_web_limit(self) -> None:
        """A web carrying compression as well as flexure has less rotation
        capacity, so it must be stockier to form a hinge."""
        light, _ = A1.lambda_pd_web(50.0, 1000.0, E_STEEL, self.FY)
        heavy, _ = A1.lambda_pd_web(500.0, 1000.0, E_STEEL, self.FY)
        assert heavy < light

    def test_the_branches_nearly_meet_at_0_125(self) -> None:
        """Eq. A-1-1 gives 3.76*(1 - 2.75*0.125) = 2.4675 and Eq. A-1-2 gives
        1.12*(2.33 - 0.125) = 2.4696 -- a 0.085% step, the Specification's own
        rounding rather than a discontinuity in intent."""
        assert pytest.approx(2.4675, abs=1e-4) == 3.76 * (1 - 2.75 * 0.125)
        assert pytest.approx(2.4696, abs=1e-4) == 1.12 * (2.33 - 0.125)
        Py = 1000.0
        Pu = 0.125 * 0.90 * Py
        below, _ = A1.lambda_pd_web(Pu * 0.9999, Py, E_STEEL, self.FY)
        above, _ = A1.lambda_pd_web(Pu * 1.0001, Py, E_STEEL, self.FY)
        assert abs(above - below) / below < 3e-3

    def test_the_floor_is_the_axial_nonslender_limit(self) -> None:
        """Eq. A-1-2's 1.49*sqrt(E/Fy) floor is Table B4.1a case 5 -- however
        much axial load the member carries, the web need never be stockier than
        the nonslender limit for pure compression."""
        limit, note = A1.lambda_pd_web(900.0, 1000.0, E_STEEL, self.FY)
        assert limit == pytest.approx(1.49 * math.sqrt(E_STEEL / self.FY), rel=1e-12)
        assert "B4.1a" in note

    def test_hss_flange_is_tighter_than_Table_B4_1b(self) -> None:
        """Eq. A-1-3's 0.94 against Table B4.1b case 17's lambda_p of 1.12."""
        got = A1.lambda_pd_hss_flange(E_STEEL, self.FY)
        assert got == pytest.approx(0.94 * math.sqrt(E_STEEL / self.FY), rel=1e-12)
        assert got < 1.12 * math.sqrt(E_STEEL / self.FY)

    def test_round_hss_has_no_square_root(self) -> None:
        """Eq. A-1-4: 0.045*E/Fy, a plain ratio -- and tighter than Table B4.1b
        case 20's 0.07*E/Fy."""
        got = A1.lambda_pd_round_hss(E_STEEL, self.FY)
        assert got == pytest.approx(0.045 * E_STEEL / self.FY, rel=1e-12)
        assert got < 0.07 * E_STEEL / self.FY

    def test_effective_end_moment_branches(self) -> None:
        """Eqs. A-1-6b and A-1-6c."""
        flat, note = A1.effective_end_moment(M1=50.0, M2=100.0, Mmid=60.0)
        assert flat == 50.0
        assert "A-1-6b" in note
        bulged, note = A1.effective_end_moment(M1=50.0, M2=100.0, Mmid=80.0)
        assert bulged == pytest.approx(60.0, rel=1e-12)
        assert "A-1-6c" in note

    def test_a_large_midspan_moment_forces_near_uniform_treatment(self) -> None:
        """Eq. A-1-6c is capped at M2: a big bulge means the segment behaves as
        if in near-uniform moment -- the worst case for LTB, correctly so."""
        capped, _ = A1.effective_end_moment(M1=50.0, M2=100.0, Mmid=200.0)
        assert capped == pytest.approx(100.0, rel=1e-12)

    def test_Lpd_uses_E_over_Fy_not_its_square_root(self) -> None:
        """Eq. A-1-5 differs from Eq. F2-5's Lp in both the gradient term and
        the unsquare-rooted E/Fy."""
        got = A1.Lpd_I_shape(0.0, 100.0, 1.65, E_STEEL, self.FY)
        assert got == pytest.approx(0.12 * (E_STEEL / self.FY) * 1.65, rel=1e-12)

    def test_uniform_moment_needs_much_closer_bracing(self) -> None:
        """The bracket runs from 0.044 at M1'/M2 = +1 to 0.196 at -1."""
        uniform = A1.Lpd_I_shape(100.0, 100.0, 1.65, E_STEEL, self.FY)
        reverse = A1.Lpd_I_shape(-100.0, 100.0, 1.65, E_STEEL, self.FY)
        assert reverse / uniform == pytest.approx(0.196 / 0.044, rel=1e-3)

    def test_box_sections_tolerate_wider_bracing_and_have_a_floor(self) -> None:
        """Eq. A-1-7 has a larger intercept AND a 0.10*(E/Fy)*ry floor that
        Eq. A-1-5 lacks -- a closed section has far more torsional stiffness."""
        i_shape = A1.Lpd_I_shape(100.0, 100.0, 1.65, E_STEEL, self.FY)
        box = A1.Lpd_bar_or_box(100.0, 100.0, 1.65, E_STEEL, self.FY)
        assert box > i_shape
        assert box == pytest.approx(0.10 * (E_STEEL / self.FY) * 1.65, rel=1e-12)

    def test_compression_bracing_reuses_the_Sect_E3_transition(self) -> None:
        """The same 4.71 -- a member with a hinge must be on the inelastic side
        of Sect. E3's boundary, because an elastically buckling column cannot
        form one."""
        from pyaisc360 import utils

        assert A1.compression_bracing_limit(3.7, E_STEEL, self.FY) == pytest.approx(
            3.7 * utils.limiting_ratio(4.71, E_STEEL, self.FY), rel=1e-12
        )

    def test_inelastic_analysis_is_LRFD_only_below_65_ksi(self) -> None:
        """Sect. 1.3.1: Fy <= 65 ksi, and Sect. 1.1 permits LRFD only."""
        assert A1.check_inelastic_limitations(50.0) == []
        assert len(A1.check_inelastic_limitations(70.0)) == 1
        assert len(A1.check_inelastic_limitations(50.0, lrfd=False)) == 1


# ===========================================================================
# Appendix 2 -- Design for Ponding
# ===========================================================================
class TestAppendix2:
    def test_Cp_and_Cs_are_dimensional_in_FEET(self) -> None:
        """Eqs. A-2-3/A-2-4 take lengths in FEET and inertias in in.^4; the 10^7
        divisor reconciles them. Passing inches understates Cp badly and the
        check then passes trivially."""
        in_feet = A2.Cp_primary(Ls=30.0, Lp=40.0, Ip=2000.0)
        in_inches = A2.Cp_primary(Ls=360.0, Lp=480.0, Ip=2000.0)
        assert in_inches / in_feet == pytest.approx(12.0**5, rel=1e-12)

    def test_Cp_value(self) -> None:
        """32*30*40^4/(10^7*2000) = 0.12288. Source: AISC Eq. A-2-3."""
        assert A2.Cp_primary(30.0, 40.0, 2000.0) == pytest.approx(0.12288, rel=1e-9)

    def test_primary_length_dominates_to_the_fourth_power(self) -> None:
        """Doubling a girder span makes the roof sixteen times more flexible."""
        a = A2.Cp_primary(30.0, 20.0, 2000.0)
        b = A2.Cp_primary(30.0, 40.0, 2000.0)
        assert b / a == pytest.approx(16.0, rel=1e-12)

    def test_the_0_9_weighting_on_Cs(self) -> None:
        """Eq. A-2-1's 0.9 is not a rounding -- secondary members deflect within
        the primary members' deflection, so they contribute slightly less than
        proportionally."""
        check = A2.simplified_ponding_check(30.0, 40.0, 6.0, 2000.0, 300.0, 0.05)
        assert check.combined == pytest.approx(check.Cp + 0.9 * check.Cs, rel=1e-12)

    def test_a_stiff_roof_is_stable(self) -> None:
        check = A2.simplified_ponding_check(
            Ls=25.0, Lp=30.0, S=5.0, Ip=4000.0, Is=800.0, Id=0.05
        )
        assert check.stable

    def test_a_flexible_roof_fails_the_stiffness_test(self) -> None:
        check = A2.simplified_ponding_check(
            Ls=40.0, Lp=50.0, S=8.0, Ip=1000.0, Is=200.0, Id=0.5
        )
        assert not check.stable
        assert check.combined > A2.PONDING_LIMIT

    def test_the_deck_check_is_separate_from_the_stiffness_check(self) -> None:
        """Eqs. A-2-1 and A-2-2 must BOTH hold; a stiff frame with a floppy deck
        still ponds."""
        check = A2.simplified_ponding_check(
            Ls=25.0, Lp=30.0, S=5.0, Ip=4000.0, Is=800.0, Id=0.001
        )
        assert check.combined <= A2.PONDING_LIMIT
        assert not check.deck_adequate
        assert not check.stable

    def test_minimum_deck_inertia(self) -> None:
        """Eq. A-2-2: Id >= 25*S^4*10^-6. At S = 6 ft that is 0.0324 in.^4/ft."""
        assert A2.minimum_deck_inertia(6.0) == pytest.approx(0.0324, rel=1e-9)

    def test_stress_index(self) -> None:
        """Eqs. A-2-5/A-2-6: U = (0.8*Fy - fo)/fo."""
        assert A2.stress_index(50.0, 20.0) == pytest.approx(1.0, rel=1e-12)

    def test_no_reserve_stress_is_rejected(self) -> None:
        """At fo >= 0.8*Fy no amount of stiffness accommodates the ponding."""
        with pytest.raises(GeometryError):
            A2.stress_index(50.0, 40.0)

    def test_report_renders(self) -> None:
        text = A2.simplified_ponding_check(30.0, 40.0, 6.0, 2000.0, 300.0, 0.05).report()
        assert "A-2-3" in text
        assert "A-2-1" in text


# ===========================================================================
# Appendix 3 -- Design for Fatigue
# ===========================================================================
class TestAppendix3:
    def test_category_A_at_one_million_cycles(self) -> None:
        """Eq. A-3-1: FSR = 1000*(25/1e6)^0.333 = 29.4 ksi.

        Source: AISC Table A-3.1 (Cf = 25) with Eq. A-3-1's leading 1,000. The
        published category A curve passes through about 29 ksi at one million
        cycles, so a result in the tens of ksi is the sanity check that the
        table and the equation are from the same edition.
        """
        FSR, _ = A3.allowable_stress_range(A3.StressCategory.A, 1.0e6)
        assert pytest.approx(1000.0 * math.pow(25.0 / 1.0e6, 0.333), rel=1e-12) == FSR
        assert pytest.approx(29.4, abs=0.5) == FSR

    def test_the_360_10_constants_would_be_off_by_a_billion(self) -> None:
        """Regression guard. 360-10 tabulated Cf = 250x10^8 and wrote Eq. A-3-1
        with no leading coefficient; 360-16 moved 10^9 into the equation. Using
        the old table with the new equation returns ~29,000 ksi -- every fatigue
        check then passes and nothing else in the library notices."""
        Cf, _ = A3.TABLE_A3_1[A3.StressCategory.A]
        assert Cf == 25.0
        assert 1000.0 * math.pow(250.0e8 / 1.0e6, 0.333) > 29_000.0

    def test_category_C_matches_the_published_curve(self) -> None:
        """Source: AISC Table A-3.1 (Cf = 4.4). About 16.4 ksi at one million
        cycles."""
        FSR, _ = A3.allowable_stress_range(A3.StressCategory.C, 1.0e6)
        assert pytest.approx(16.4, abs=0.3) == FSR

    def test_every_curve_crosses_its_own_threshold(self) -> None:
        """A category whose finite-life curve never reached FTH would have no
        indefinite-life regime at all. Each must cross within the practical
        10^4 to 10^9 cycle window."""
        for category, (_, FTH) in A3.TABLE_A3_1.items():
            at_1e4, _ = A3.allowable_stress_range(category, 1.0e4)
            at_1e9, _ = A3.allowable_stress_range(category, 1.0e9)
            assert at_1e4 > FTH
            assert at_1e9 == pytest.approx(FTH, rel=1e-12)

    def test_the_threshold_makes_indefinite_life_possible(self) -> None:
        """Past the cycle count at which the curve drops below FTH, the
        allowable stress range stops falling."""
        FSR, note = A3.allowable_stress_range(A3.StressCategory.A, 1.0e12)
        assert pytest.approx(24.0, rel=1e-12) == FSR
        assert "threshold" in note.lower()

    @pytest.mark.parametrize(
        ("category", "Cf", "FTH"),
        [
            (A3.StressCategory.A, 25.0, 24.0),
            (A3.StressCategory.B, 12.0, 16.0),
            (A3.StressCategory.B_PRIME, 6.1, 12.0),
            (A3.StressCategory.C, 4.4, 10.0),
            (A3.StressCategory.D, 2.2, 7.0),
            (A3.StressCategory.E, 1.1, 4.5),
            (A3.StressCategory.E_PRIME, 0.39, 2.6),
        ],
    )
    def test_published_table_values(
        self, category: A3.StressCategory, Cf: float, FTH: float
    ) -> None:
        """Source: AISC Table A-3.1 -- printed constants, asserted exactly."""
        assert A3.TABLE_A3_1[category] == (Cf, FTH)

    def test_the_categories_are_monotonically_weaker(self) -> None:
        order = [
            A3.StressCategory.A,
            A3.StressCategory.B,
            A3.StressCategory.B_PRIME,
            A3.StressCategory.C,
            A3.StressCategory.D,
            A3.StressCategory.E,
            A3.StressCategory.E_PRIME,
        ]
        values = [A3.allowable_stress_range(c, 1.0e6)[0] for c in order]
        assert all(a > b for a, b in zip(values, values[1:], strict=False))

    def test_category_F_has_a_much_flatter_slope(self) -> None:
        """Eq. A-3-2's 0.167 exponent against 0.333 everywhere else -- weld shear
        degrades far more slowly with cycle count than base-metal tension."""
        a = A3.category_F_stress_range(1.0e5)[0]
        b = A3.category_F_stress_range(1.0e6)[0]
        assert a / b == pytest.approx(math.pow(10.0, 0.167), rel=1e-12)
        c = A3.allowable_stress_range(A3.StressCategory.C, 1.0e5)[0]
        d = A3.allowable_stress_range(A3.StressCategory.C, 1.0e6)[0]
        assert c / d == pytest.approx(math.pow(10.0, 0.333), rel=1e-12)

    def test_category_F_threshold_is_8_ksi(self) -> None:
        assert A3.category_F_stress_range(1.0e12)[0] == pytest.approx(8.0, rel=1e-12)

    def test_category_F_rejects_Eq_A_3_1(self) -> None:
        with pytest.raises(AISC360Error):
            A3.allowable_stress_range(A3.StressCategory.F, 1.0e6)

    def test_the_root_equations_have_no_threshold(self) -> None:
        """Eqs. A-3-3/A-3-5 keep falling for ever -- a root-initiated crack has
        no endurance limit, which is why the weld toe must be checked as
        category C as well."""
        FSR, note = A3.PJP_root_stress_range(0.8, 1.0e12)
        assert FSR < 1.0
        assert "threshold" in note.lower()

    def test_reduction_factors_are_capped_at_unity(self) -> None:
        """R = 1.0 means the root is no worse than the toe."""
        assert A3.PJP_reduction_factor(0.0, 1.0, 0.5) == pytest.approx(1.0, rel=1e-12)
        assert A3.fillet_reduction_factor(1.0, 0.5) == pytest.approx(1.0, rel=1e-12)

    def test_a_deeper_unwelded_root_reduces_RPJP(self) -> None:
        shallow = A3.PJP_reduction_factor(0.2, 0.25, 1.0)
        deep = A3.PJP_reduction_factor(0.6, 0.25, 1.0)
        assert deep < shallow

    def test_bolt_tensile_area_is_not_the_body_area(self) -> None:
        """Eq. A-3-7's (db - 0.9743/n) thread deduction. A 3/4-in. UNC bolt gives
        0.334 in.^2 against a 0.442 in.^2 body area -- 24% less. Chapter J puts
        the thread reduction inside Fn instead, and mixing the two conventions
        either double-counts it or drops it."""
        At = A3.bolt_tensile_area(0.75, 10.0)
        body = math.pi * 0.75**2 / 4.0
        assert At == pytest.approx(0.334, abs=0.002)
        assert body == pytest.approx(0.442, abs=0.002)

    def test_fatigue_may_be_disregarded_below_the_threshold(self) -> None:
        ok, _ = A3.fatigue_need_not_be_considered(A3.StressCategory.C, 8.0)
        assert ok
        no, _ = A3.fatigue_need_not_be_considered(A3.StressCategory.C, 12.0)
        assert not no


# ===========================================================================
# Appendix 4 -- Structural Design for Fire Conditions
# ===========================================================================
class TestAppendix4:
    def test_below_400F_nothing_changes(self) -> None:
        """Sect. 4.2.4d: at or below 400 F the design strengths are determined
        without consideration of temperature effects. A hard cut-off, not a
        taper."""
        props = A4.steel_properties(400.0, 50.0)
        assert not props.reduced
        assert (props.kE, props.ky) == (1.0, 1.0)
        assert props.Fy == pytest.approx(50.0, rel=1e-12)

    def test_above_400F_the_reductions_begin(self) -> None:
        props = A4.steel_properties(1000.0, 50.0)
        assert props.reduced
        assert props.kE == pytest.approx(0.49, rel=1e-12)
        assert props.ky == pytest.approx(0.66, rel=1e-12)
        assert props.Fy == pytest.approx(33.0, rel=1e-12)

    def test_ku_is_normalised_on_Fy_not_Fu(self) -> None:
        """Table A-4.2.1's ku is Fu(T)/**Fy**. Reading it as a fraction of Fu
        overstates Fu(T) by the Fu/Fy ratio -- about 30% for A992."""
        props = A4.steel_properties(1000.0, 50.0, Fu=65.0)
        assert props.Fu == pytest.approx(props.ku * 50.0, rel=1e-12)
        assert props.Fu != pytest.approx(props.ku * 65.0, rel=1e-3)

    def test_stiffness_degrades_faster_than_strength(self) -> None:
        """At 1000 F, kE = 0.49 against ky = 0.66 -- steel goes soft before it
        goes weak, which is what drives fire deflections."""
        props = A4.steel_properties(1000.0, 50.0)
        assert props.kE < props.ky

    def test_the_tables_interpolate_linearly(self) -> None:
        """1100 F sits midway between the 1000 and 1200 rows."""
        low = A4.steel_properties(1000.0, 50.0).kE
        high = A4.steel_properties(1200.0, 50.0).kE
        mid = A4.steel_properties(1100.0, 50.0).kE
        assert mid == pytest.approx((low + high) / 2.0, rel=1e-12)

    def test_beyond_the_table_is_out_of_scope(self) -> None:
        """Refusing to extrapolate past 2200 F: the retention factors are
        already at 0.02 and a linear continuation goes negative."""
        with pytest.raises(OutOfScopeError):
            A4.steel_properties(2400.0, 50.0)

    def test_lightweight_concrete_holds_up_far_better(self) -> None:
        """Table A-4.2.2 at 550 F: kc = 1.00 lightweight against 0.86 normal
        weight -- the expanded aggregate has already been fired."""
        nw, _, _ = A4.concrete_properties(550.0, lightweight=False)
        lw, _, _ = A4.concrete_properties(550.0, lightweight=True)
        assert lw > nw

    def test_concrete_stiffness_falls_much_faster_than_strength(self) -> None:
        """The same pattern as steel, but far more pronounced."""
        kc, kEc, _ = A4.concrete_properties(1000.0)
        assert kEc < kc

    def test_bolts_degrade_earlier_than_the_steel_they_connect(self) -> None:
        """Table A-4.2.3 gives 0.93 at 400 F where Table A-4.2.1 still gives
        ky = 1.00. A connection can become the weak link in a fire."""
        assert A4.bolt_retention(400.0) < 1.0
        assert A4.steel_properties(400.0, 50.0).ky == 1.0
        assert A4.bolt_retention(1000.0) < A4.steel_properties(1000.0, 50.0).ky

    def test_the_fire_load_combination(self) -> None:
        """Eq. A-4-1: (0.9 or 1.2)*D + AT + 0.5*L + 0.2*S, with AT UNFACTORED --
        the fire itself carries no load factor."""
        assert A4.fire_load_combination(100.0, 40.0, 50.0, 20.0) == pytest.approx(
            1.2 * 100 + 40 + 0.5 * 50 + 0.2 * 20, rel=1e-12
        )
        assert A4.fire_load_combination(
            100.0, 40.0, 50.0, 20.0, uplift=True
        ) == pytest.approx(0.9 * 100 + 40 + 25 + 4, rel=1e-12)

    def test_Eq_A_4_2_uses_0_42_and_a_square_root(self) -> None:
        """Two differences from Eq. E3-2: the base is 0.42 not 0.658, and the
        exponent carries a square root Eq. E3-2 does not have. Hot steel has no
        distinct elastic/inelastic transition, so one branch covers everything
        and the result is always below the ambient curve."""
        from pyaisc360 import utils

        Fy_T, Fe_T = 33.0, 40.0
        got = A4.Fcr_at_temperature(Fy_T, Fe_T)
        assert got == pytest.approx(
            math.pow(0.42, math.sqrt(Fy_T / Fe_T)) * Fy_T, rel=1e-12
        )
        assert got < utils.flexural_buckling_stress(Fy_T, Fe_T)

    def test_FL_at_ambient_recovers_Eq_F4_6a(self) -> None:
        """Eq. A-4-8 with kp = ky = 1.0 gives 0.7*Fy, exactly Eq. F4-6a."""
        assert A4.FL_at_temperature(50.0, 1.0, 1.0) == pytest.approx(35.0, rel=1e-12)

    def test_cx_grows_with_temperature_and_caps_at_3(self) -> None:
        """Eq. A-4-10: cx = 0.53 + T/450 <= 3.0."""
        assert A4.cx_coefficient(68.0) == pytest.approx(0.53 + 68.0 / 450.0, rel=1e-12)
        assert A4.cx_coefficient(2000.0) == pytest.approx(3.0, rel=1e-12)

    def test_the_LTB_curve_has_no_Lp_plateau(self) -> None:
        """Eq. A-4-3 starts falling from Lb = 0 -- a hot beam has no plastic
        plateau, and the variable is (Lb/Lr(T))^cx, not a linear fraction."""
        Mn, _ = A4.Mn_ltb_at_temperature(
            Lb=1.0, Lr_T=200.0, Mp_T=3000.0, Mr_T=2000.0, cx=1.5
        )
        assert Mn < 3000.0

    def test_the_branches_meet_at_Lr_T(self) -> None:
        """At Lb = Lr(T) the bracket gives Mp - (Mp - Mr) = Mr exactly."""
        Mn, _ = A4.Mn_ltb_at_temperature(
            Lb=200.0, Lr_T=200.0, Mp_T=3000.0, Mr_T=2000.0, cx=1.5
        )
        assert Mn == pytest.approx(2000.0, rel=1e-12)

    def test_the_LTB_result_is_capped_at_Mp_T(self) -> None:
        Mn, _ = A4.Mn_ltb_at_temperature(
            Lb=10.0, Lr_T=200.0, Mp_T=3000.0, Mr_T=2000.0, cx=1.5, Cb=3.0
        )
        assert Mn == pytest.approx(3000.0, rel=1e-12)

    def test_composite_scales_by_the_bottom_flange_retention(self) -> None:
        """Eq. A-4-11: Mn(T) = r(T)*Mn, indexed on the bottom flange -- both the
        hottest point and the one carrying the tension."""
        assert A4.composite_flexural_at_temperature(0.6, 5000.0) == pytest.approx(
            3000.0, rel=1e-12
        )

    def test_a_retention_factor_outside_the_unit_interval_is_rejected(self) -> None:
        with pytest.raises(GeometryError):
            A4.composite_flexural_at_temperature(1.5, 5000.0)
