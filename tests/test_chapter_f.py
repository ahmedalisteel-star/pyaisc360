"""Chapter F -- Design of Members for Flexure.

Precision policy, per the two things being tested:

* **Catalog-rounded quantities** -- anything traceable to a published table
  (Lp, Lr, phi*Mp against AISC *Manual* Table 3-2) is asserted at 0.5%, the
  table's own three-significant-figure precision.
* **Internal mathematical transitions** -- Cb linearity, the Mp cap, branch
  continuity at Lb = Lp, FL for a doubly symmetric shape, Rpc at a compact web
  -- are asserted at 1e-12. These are exact identities in the algebra, and
  anything looser would hide a real error.

One transition is deliberately *not* held to 1e-12: Eqs. F2-2 and F2-3 do not
meet at Lb = Lr, because both coefficients in Eq. F2-6 are rounded. That is the
Specification's own discontinuity, measured and pinned in
:class:`TestF2BranchContinuity`.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pyaisc360 import chapter_f as F
from pyaisc360.chapter_b import FlexuralElement, flexural_limits
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import Axis, FlexuralSlenderness, LimitState
from pyaisc360.core.exceptions import GeometryError, OutOfScopeError
from pyaisc360.materials import A36, A500_C_RECT, A992
from pyaisc360.sections import SectionAdapter

E_STEEL = 29000.0
FY50 = 50.0
COMPACT = FlexuralSlenderness.COMPACT
NONCOMPACT = FlexuralSlenderness.NONCOMPACT
SLENDER = FlexuralSlenderness.SLENDER

#: AISC Shapes Database v15.0 properties for the benchmark shapes. Test data,
#: not a shipped catalog -- pyaisc360 has none.
SHAPES: dict[str, dict[str, float]] = {
    "W18X50": dict(Ag=14.7, d=18.0, tw=0.355, bf=7.50, tf=0.570, Ix=800.0, Zx=101.0,
                   Sx=88.9, rx=7.38, Iy=40.1, Zy=19.0, Sy=10.7, ry=1.65, J=1.24,
                   rts=1.98, ho=17.4, bf_2tf=6.57, h_tw=45.2),
    "W16X26": dict(Ag=7.68, d=15.7, tw=0.250, bf=5.50, tf=0.345, Ix=301.0, Zx=44.2,
                   Sx=38.4, rx=6.26, Iy=9.59, Zy=5.48, Sy=3.49, ry=1.12, J=0.262,
                   rts=1.38, ho=15.4, bf_2tf=7.97, h_tw=56.8),
    "W24X68": dict(Ag=20.1, d=23.7, tw=0.415, bf=8.97, tf=0.585, Ix=1830.0, Zx=177.0,
                   Sx=154.0, rx=9.55, Iy=70.4, Zy=24.5, Sy=15.7, ry=1.87, J=1.87,
                   rts=2.26, ho=23.1, bf_2tf=7.66, h_tw=52.0),
    "W24X55": dict(Ag=16.2, d=23.6, tw=0.395, bf=7.01, tf=0.505, Ix=1350.0, Zx=134.0,
                   Sx=114.0, rx=9.11, Iy=29.1, Zy=13.3, Sy=8.30, ry=1.34, J=1.18,
                   rts=1.68, ho=23.1, bf_2tf=6.94, h_tw=54.6),
}


def shape(name: str) -> SectionAdapter:
    return SectionAdapter.from_mapping({"name": name, **SHAPES[name]})


@pytest.fixture
def w18x50() -> SectionAdapter:
    return shape("W18X50")


def compact_member(section: SectionAdapter, **kwargs: Any) -> F.FlexuralMember:
    return F.FlexuralMember(section, A992, shape=F.ShapeType.ROLLED_I, **kwargs)


# ===========================================================================
# Table B4.1b
# ===========================================================================
class TestTableB4_1b:
    @pytest.mark.parametrize(
        ("element", "lam_p", "lam_r"),
        [
            (FlexuralElement.ROLLED_I_FLANGE, 9.1516, 24.0832),  # case 10: 0.38 / 1.0
            (FlexuralElement.ANGLE_LEG, 13.0049, 21.9157),  # case 12: 0.54 / 0.91
            (FlexuralElement.MINOR_AXIS_FLANGE, 9.1516, 24.0832),  # case 13
            (FlexuralElement.TEE_STEM, 20.2299, 36.6064),  # case 14: 0.84 / 1.52
            (FlexuralElement.I_WEB, 90.5528, 137.2742),  # case 15: 3.76 / 5.70
            (FlexuralElement.RECTANGULAR_HSS_FLANGE, 26.9732, 33.7165),  # case 17
            (FlexuralElement.COVER_PLATE, 26.9732, 33.7165),  # case 18
            (FlexuralElement.RECTANGULAR_HSS_WEB, 58.2813, 137.2742),  # case 19
            (FlexuralElement.BOX_FLANGE, 26.9732, 35.8840),  # case 21: 1.12 / 1.49
        ],
    )
    def test_limits_at_fy_50(
        self, element: FlexuralElement, lam_p: float, lam_r: float
    ) -> None:
        got_p, got_r = flexural_limits(element, E_STEEL, FY50)
        assert got_p == pytest.approx(lam_p, abs=1e-4)
        assert got_r == pytest.approx(lam_r, abs=1e-4)

    def test_round_hss_has_no_square_root(self) -> None:
        """Case 20 is 0.07*E/Fy and 0.31*E/Fy -- plain ratios, as in case 9."""
        lam_p, lam_r = flexural_limits(FlexuralElement.ROUND_HSS, E_STEEL, FY50)
        assert lam_p == pytest.approx(0.07 * E_STEEL / FY50, rel=1e-12)
        assert lam_r == pytest.approx(0.31 * E_STEEL / FY50, rel=1e-12)
        assert (lam_p, lam_r) == pytest.approx((40.6, 179.8), abs=0.05)

    def test_built_up_flange_uses_FL_not_Fy(self) -> None:
        """Case 11 footnote [b]: lambda_r = 0.95*sqrt(kc*E/FL).

        Using Fy where FL belongs overstates lambda_r -- at FL = 0.7*Fy the
        error is sqrt(1/0.7) = 1.195, i.e. 20% high, which would call a slender
        flange noncompact.
        """
        _, with_FL = flexural_limits(
            FlexuralElement.BUILT_UP_I_FLANGE, E_STEEL, FY50, h_over_tw=100.0, FL=35.0
        )
        _, with_Fy = flexural_limits(
            FlexuralElement.BUILT_UP_I_FLANGE, E_STEEL, FY50, h_over_tw=100.0, FL=50.0
        )
        assert with_FL / with_Fy == pytest.approx(math.sqrt(50.0 / 35.0), rel=1e-12)

    def test_built_up_flange_requires_its_arguments(self) -> None:
        with pytest.raises(Exception, match="h_over_tw"):
            flexural_limits(FlexuralElement.BUILT_UP_I_FLANGE, E_STEEL, FY50)

    def test_singly_symmetric_web_case_16(self) -> None:
        """lambda_p = (hc/hp)*sqrt(E/Fy)/(0.54*Mp/My - 0.09)^2, capped at lambda_r."""
        lam_p, lam_r = flexural_limits(
            FlexuralElement.SINGLY_SYMMETRIC_I_WEB, E_STEEL, FY50,
            Mp_over_My=1.12, hc_over_hp=1.0,
        )
        expected = math.sqrt(E_STEEL / FY50) / (0.54 * 1.12 - 0.09) ** 2
        assert lam_p == pytest.approx(min(expected, lam_r), rel=1e-12)
        assert lam_r == pytest.approx(5.70 * math.sqrt(E_STEEL / FY50), rel=1e-12)

    def test_case_16_is_capped_at_lambda_r(self) -> None:
        """Footnote [c]: lambda_p <= lambda_r. A low shape factor blows up the
        denominator's reciprocal, and without the cap lambda_p would exceed
        lambda_r -- making a slender web classify as compact."""
        lam_p, lam_r = flexural_limits(
            FlexuralElement.SINGLY_SYMMETRIC_I_WEB, E_STEEL, FY50,
            Mp_over_My=1.10, hc_over_hp=1.5,
        )
        assert lam_p == lam_r

    def test_case_16_rejects_a_degenerate_shape_factor(self) -> None:
        with pytest.raises(Exception, match="not positive"):
            flexural_limits(
                FlexuralElement.SINGLY_SYMMETRIC_I_WEB, E_STEEL, FY50,
                Mp_over_My=0.15, hc_over_hp=1.0,
            )


# ===========================================================================
# Sect. F2
# ===========================================================================
class TestF2:
    def test_Lp_w18x50(self) -> None:
        """Eq. F2-5: Lp = 1.76*1.65*sqrt(29000/50) = 69.94 in = 5.83 ft."""
        assert F.Lp_F2(1.65, E_STEEL, FY50) == pytest.approx(69.938, abs=0.01)

    def test_Lr_w18x50(self) -> None:
        """Eq. F2-6 for W18x50 -> 203.35 in = 16.95 ft (Table 3-2 prints 16.9)."""
        Lr = F.Lr_F2(1.98, E_STEEL, FY50, 1.24, 1.0, 88.9, 17.4)
        assert Lr / 12.0 == pytest.approx(16.9, rel=0.005)

    def test_Mp_is_Fy_times_Zx(self) -> None:
        """Eq. F2-1 has no 1.6*Fy*S cap -- unlike Eqs. F6-1, F9-2 and F11-1."""
        assert F.plastic_moment(FY50, 101.0) == pytest.approx(5050.0, rel=1e-12)

    def test_ltb_does_not_apply_below_Lp(self, w18x50: SectionAdapter) -> None:
        """Sect. F2.2(a): only the yielding state is returned."""
        states = F.f2_strength(compact_member(w18x50, Lb=60.0))
        assert [s.limit_state for s in states] == [LimitState.PLASTIC_MOMENT]

    def test_inelastic_ltb_between_Lp_and_Lr(self, w18x50: SectionAdapter) -> None:
        states = F.f2_strength(compact_member(w18x50, Lb=120.0))
        ltb = next(s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING)
        assert ltb.citation.equation == "F2-2"
        assert ltb.nominal < 5050.0

    def test_elastic_ltb_beyond_Lr(self, w18x50: SectionAdapter) -> None:
        states = F.f2_strength(compact_member(w18x50, Lb=300.0))
        ltb = next(s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING)
        assert ltb.citation.equation == "F2-3"

    def test_strength_decreases_monotonically_with_Lb(self, w18x50: SectionAdapter) -> None:
        strengths = [
            F.flexural_strength(
                compact_member(w18x50, Lb=float(Lb)), flange_class=COMPACT, web_class=COMPACT
            ).nominal
            for Lb in range(12, 480, 12)
        ]
        assert all(a >= b for a, b in zip(strengths, strengths[1:], strict=False))

    def test_Fcr_uses_the_warping_term(self) -> None:
        """The User Note permits the sqrt term to be 1.0; evaluating it is
        worth a substantial margin on a stocky rolled shape."""
        with_warping = F.Fcr_F2(1.0, 300.0, 1.98, E_STEEL, 1.24, 1.0, 88.9, 17.4)
        without = 1.0 * math.pi**2 * E_STEEL / (300.0 / 1.98) ** 2
        assert with_warping > without
        # sqrt(1 + 0.078*(J*c/(Sx*ho))*(Lb/rts)^2) = sqrt(2.435) = 1.561
        assert with_warping / without == pytest.approx(1.561, rel=0.01)

    def test_channel_c_below_unity_reduces_Lr(self) -> None:
        """Eq. F2-8b gives c < 1 for a channel, which shortens Lr."""
        Lr_doubly = F.Lr_F2(1.98, E_STEEL, FY50, 1.24, 1.0, 88.9, 17.4)
        Lr_channel = F.Lr_F2(1.98, E_STEEL, FY50, 1.24, 0.85, 88.9, 17.4)
        assert Lr_channel < Lr_doubly


class TestF2BranchContinuity:
    """Exact identities at the branch boundaries -- and the one that is not exact."""

    SX, ZX, RY, RTS, J, HO = 88.9, 101.0, 1.65, 1.98, 1.24, 17.4

    def test_F2_2_returns_Mp_exactly_at_Lp(self, w18x50: SectionAdapter) -> None:
        """At Lb = Lp the interpolation fraction is zero, so Eq. F2-2 gives Cb*Mp,
        and the cap brings it to Mp. Exact -- asserted at 1e-12."""
        Lp = F.Lp_F2(self.RY, E_STEEL, FY50)
        states = F.f2_strength(compact_member(w18x50, Lb=Lp * (1 + 1e-15)))
        Mp = FY50 * self.ZX
        ltb = [s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING]
        if ltb:
            assert ltb[0].nominal == pytest.approx(Mp, rel=1e-12)

    def test_the_F2_6_coefficients_are_rounded(self) -> None:
        """Eq. F2-6 inverts Eq. F2-4 at Fcr = 0.7*Fy, so its coefficients are
        derivable -- and both are printed rounded::

            1.95 vs sqrt(pi^4*0.078/2) = 1.949091   (0.047% high)
            6.76 vs 4/(pi^4*0.078^2)   = 6.749495   (0.156% high)
        """
        assert math.sqrt(math.pi**4 * 0.078 / 2.0) == pytest.approx(1.949091, abs=1e-6)
        assert 4.0 / (math.pi**4 * 0.078**2) == pytest.approx(6.749495, abs=1e-6)

    def test_exact_coefficients_make_Fcr_at_Lr_exactly_0_7_Fy(self) -> None:
        """With the unrounded coefficients the round trip closes to 2e-16."""
        k = self.J * 1.0 / (self.SX * self.HO)
        beta = 0.7 * FY50 / E_STEEL
        lead = math.sqrt(math.pi**4 * 0.078 / 2.0)
        resid = 4.0 / (math.pi**4 * 0.078**2)
        Lr_exact = lead * self.RTS / beta * math.sqrt(k + math.sqrt(k * k + resid * beta**2))
        Fcr = F.Fcr_F2(1.0, Lr_exact, self.RTS, E_STEEL, self.J, 1.0, self.SX, self.HO)
        assert Fcr == pytest.approx(0.7 * FY50, rel=1e-12)

    def test_printed_coefficients_leave_a_measurable_step_at_Lr(self) -> None:
        """With the printed coefficients, Eqs. F2-2 and F2-3 differ by 0.12% at Lr.

        Pinned rather than smoothed: the printed values are what the
        Specification mandates and what Manual Table 3-2's Lr reflects.
        """
        Lr = F.Lr_F2(self.RTS, E_STEEL, FY50, self.J, 1.0, self.SX, self.HO)
        Fcr = F.Fcr_F2(1.0, Lr, self.RTS, E_STEEL, self.J, 1.0, self.SX, self.HO)
        by_F2_3 = Fcr * self.SX
        by_F2_2 = 0.7 * FY50 * self.SX  # Eq. F2-2 at Lb = Lr
        step = abs(by_F2_2 - by_F2_3) / by_F2_2
        assert 5e-4 < step < 3e-3


class TestCbIntegration:
    """Cb must act as a clean linear multiplier, then be capped at Mp."""

    def test_cb_scales_the_inelastic_branch_linearly(self, w18x50: SectionAdapter) -> None:
        """Below the cap, doubling Cb doubles Mn -- exactly."""
        base = F.f2_strength(compact_member(w18x50, Lb=260.0, Cb=1.0))
        doubled = F.f2_strength(compact_member(w18x50, Lb=260.0, Cb=2.0))
        LTB = LimitState.LATERAL_TORSIONAL_BUCKLING
        Mn1 = next(s.nominal for s in base if s.limit_state is LTB)
        Mn2 = next(s.nominal for s in doubled if s.limit_state is LTB)
        assert Mn2 == pytest.approx(2.0 * Mn1, rel=1e-12)

    def test_cb_is_capped_at_Mp(self, w18x50: SectionAdapter) -> None:
        """A large Cb must never produce Mn above the plastic moment."""
        Mp = FY50 * 101.0
        for Cb in (1.0, 1.67, 2.27, 3.0, 10.0):
            result = F.flexural_strength(
                compact_member(w18x50, Lb=100.0, Cb=Cb), flange_class=COMPACT, web_class=COMPACT
            )
            assert result.nominal <= Mp * (1 + 1e-12)

    def test_capped_ltb_helper(self) -> None:
        assert F.capped_ltb(2.0, 100.0, 5050.0) == pytest.approx(200.0, rel=1e-12)
        assert F.capped_ltb(100.0, 100.0, 5050.0) == pytest.approx(5050.0, rel=1e-12)

    def test_capped_ltb_rejects_non_positive_cb(self) -> None:
        with pytest.raises(GeometryError, match="Cb must be positive"):
            F.capped_ltb(0.0, 100.0, 5050.0)

    def test_the_published_cb_values_flow_through(self, w18x50: SectionAdapter) -> None:
        """Cb = 2.27 (User Note F1.1 reverse curvature) lifts a long-span beam."""
        from pyaisc360.utils import lateral_torsional_modification_factor

        Cb = lateral_torsional_modification_factor(100.0, 50.0, 0.0, 50.0)
        assert Cb == pytest.approx(2.2727, abs=1e-4)
        uniform = F.flexural_strength(
            compact_member(w18x50, Lb=300.0, Cb=1.0), flange_class=COMPACT, web_class=COMPACT
        ).nominal
        reversed_curvature = F.flexural_strength(
            compact_member(w18x50, Lb=300.0, Cb=Cb), flange_class=COMPACT, web_class=COMPACT
        ).nominal
        assert reversed_curvature > uniform


# ===========================================================================
# Sect. F3
# ===========================================================================
class TestF3:
    def test_noncompact_flange_interpolates(self, w18x50: SectionAdapter) -> None:
        lam_pf, lam_rf = flexural_limits(FlexuralElement.ROLLED_I_FLANGE, E_STEEL, FY50)
        states = F.f3_strength(
            compact_member(w18x50, Lb=0.0), flange_class=NONCOMPACT,
            lam=10.2, lam_pf=lam_pf, lam_rf=lam_rf, h_over_tw=25.9,
        )
        flb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert flb.citation.equation == "F3-1"
        assert 0.7 * FY50 * 88.9 < flb.nominal < FY50 * 101.0

    def test_at_lambda_pf_F3_1_returns_Mp_exactly(self, w18x50: SectionAdapter) -> None:
        lam_pf, lam_rf = flexural_limits(FlexuralElement.ROLLED_I_FLANGE, E_STEEL, FY50)
        states = F.f3_strength(
            compact_member(w18x50, Lb=0.0), flange_class=NONCOMPACT,
            lam=lam_pf, lam_pf=lam_pf, lam_rf=lam_rf, h_over_tw=25.9,
        )
        flb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert flb.nominal == pytest.approx(FY50 * 101.0, rel=1e-12)

    def test_at_lambda_rf_F3_1_returns_0_7_Fy_Sx_exactly(self, w18x50: SectionAdapter) -> None:
        lam_pf, lam_rf = flexural_limits(FlexuralElement.ROLLED_I_FLANGE, E_STEEL, FY50)
        states = F.f3_strength(
            compact_member(w18x50, Lb=0.0), flange_class=NONCOMPACT,
            lam=lam_rf, lam_pf=lam_pf, lam_rf=lam_rf, h_over_tw=25.9,
        )
        flb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert flb.nominal == pytest.approx(0.7 * FY50 * 88.9, rel=1e-12)

    def test_slender_flange_uses_kc(self, w18x50: SectionAdapter) -> None:
        lam_pf, lam_rf = flexural_limits(FlexuralElement.ROLLED_I_FLANGE, E_STEEL, FY50)
        states = F.f3_strength(
            compact_member(w18x50, Lb=0.0), flange_class=SLENDER,
            lam=30.0, lam_pf=lam_pf, lam_rf=lam_rf, h_over_tw=100.0,
        )
        flb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert flb.citation.equation == "F3-2"
        assert flb.nominal == pytest.approx(0.9 * E_STEEL * 0.4 * 88.9 / 900.0, rel=1e-12)

    def test_compact_flange_is_out_of_scope(self, w18x50: SectionAdapter) -> None:
        lam_pf, lam_rf = flexural_limits(FlexuralElement.ROLLED_I_FLANGE, E_STEEL, FY50)
        with pytest.raises(OutOfScopeError, match="Sect. F2"):
            F.f3_strength(
                compact_member(w18x50), flange_class=COMPACT,
                lam=5.0, lam_pf=lam_pf, lam_rf=lam_rf, h_over_tw=25.9,
            )


# ===========================================================================
# Sect. F4
# ===========================================================================
class TestF4Components:
    def test_FL_is_0_7_Fy_for_a_doubly_symmetric_shape(self) -> None:
        """Sxt = Sxc gives exactly 0.7*Fy -- the origin of the 0.7 in Sect. F2."""
        assert F.FL_stress(FY50, 88.9, 88.9) == pytest.approx(0.7 * FY50, rel=1e-12)

    def test_FL_switches_at_Sxt_over_Sxc_of_0_7(self) -> None:
        assert F.FL_stress(FY50, 70.0, 100.0) == pytest.approx(0.7 * FY50, rel=1e-12)
        assert F.FL_stress(FY50, 69.0, 100.0) == pytest.approx(FY50 * 0.69, rel=1e-12)

    def test_FL_has_a_0_5_Fy_floor(self) -> None:
        """Eq. F4-6b: FL = Fy*Sxt/Sxc but not below 0.5*Fy."""
        assert F.FL_stress(FY50, 20.0, 100.0) == pytest.approx(0.5 * FY50, rel=1e-12)

    def test_aw_ratio(self) -> None:
        """Eq. F4-12: aw = hc*tw/(bfc*tfc)."""
        assert F.aw_ratio(20.0, 0.375, 8.0, 0.625) == pytest.approx(
            20.0 * 0.375 / (8.0 * 0.625), rel=1e-12
        )

    def test_rt_effective_radius(self) -> None:
        """Eq. F4-11: rt = bfc/sqrt(12*(1 + aw/6))."""
        assert F.rt_effective_radius(8.0, 1.5) == pytest.approx(
            8.0 / math.sqrt(12.0 * 1.25), rel=1e-12
        )

    def test_Lp_F4_uses_1_1_not_1_76(self) -> None:
        """Eq. F4-7's coefficient is 1.1 against rt, not Eq. F2-5's 1.76 against ry."""
        assert F.Lp_F4(2.0, E_STEEL, FY50) == pytest.approx(
            1.1 * 2.0 * math.sqrt(E_STEEL / FY50), rel=1e-12
        )
        assert F.Lp_F4(2.0, E_STEEL, FY50) < F.Lp_F2(2.0, E_STEEL, FY50)


class TestWebPlastificationFactor:
    def test_Rpc_is_unity_below_the_0_23_cutoff(self) -> None:
        """Eq. F4-10: Iyc/Iy <= 0.23 takes no plastification credit at all."""
        assert F.web_plastification_factor(5050.0, 4000.0, 100.0, 90.0, 137.0, 0.20) == 1.0

    def test_Rpc_is_the_shape_factor_at_a_compact_web(self) -> None:
        """Eq. F4-9a: Rpc = Mp/Myc exactly."""
        assert F.web_plastification_factor(
            5050.0, 4445.0, 50.0, 90.0, 137.0, 0.5
        ) == pytest.approx(5050.0 / 4445.0, rel=1e-12)

    def test_Rpc_interpolates_for_a_noncompact_web(self) -> None:
        """Eq. F4-9b interpolates the shape factor down towards 1.0."""
        Rpc = F.web_plastification_factor(5050.0, 4445.0, 113.5, 90.0, 137.0, 0.5)
        assert 1.0 < Rpc < 5050.0 / 4445.0

    def test_Rpc_at_lambda_pw_equals_the_shape_factor_exactly(self) -> None:
        assert F.web_plastification_factor(
            5050.0, 4445.0, 90.0, 90.0, 137.0, 0.5
        ) == pytest.approx(5050.0 / 4445.0, rel=1e-12)

    def test_Rpc_at_lambda_rw_reaches_unity_exactly(self) -> None:
        assert F.web_plastification_factor(
            5050.0, 4445.0, 137.0, 90.0, 137.0, 0.5
        ) == pytest.approx(1.0, rel=1e-12)

    def test_Rpc_is_capped_at_the_shape_factor(self) -> None:
        assert F.web_plastification_factor(5050.0, 4445.0, 50.0, 90.0, 137.0, 0.5) <= (
            5050.0 / 4445.0
        )


class TestF4Strength:
    """A singly symmetric plate girder with a noncompact web."""

    KWARGS: dict[str, Any] = dict(
        Sxc=300.0, Sxt=340.0, hc=40.0, bfc=12.0, tfc=0.75, tw=0.375,
        Iyc_over_Iy=0.55, flange_class=COMPACT, lam_f=8.0, lam_pf=9.15, lam_rf=24.08,
        lam_w=106.7, lam_pw=90.55, lam_rw=137.27, ho=40.75,
    )

    @pytest.fixture
    def girder(self) -> SectionAdapter:
        return SectionAdapter.from_mapping(
            {"name": "PG40", "Ag": 39.0, "Zx": 360.0, "Sx": 300.0, "Iy": 216.0,
             "ry": 2.35, "J": 4.2}
        )

    def test_compression_flange_yielding_is_evaluated(self, girder: SectionAdapter) -> None:
        states = F.f4_strength(F.FlexuralMember(girder, A992, Lb=0.0), **self.KWARGS)
        cfy = next(s for s in states if s.limit_state is LimitState.COMPRESSION_FLANGE_YIELDING)
        assert cfy.citation.equation == "F4-1"
        assert cfy.nominal > FY50 * 300.0  # Rpc > 1

    def test_tension_flange_yielding_is_skipped_when_Sxt_exceeds_Sxc(
        self, girder: SectionAdapter
    ) -> None:
        """Sect. F4.4(a): when Sxt >= Sxc the limit state does not apply."""
        states = F.f4_strength(F.FlexuralMember(girder, A992, Lb=0.0), **self.KWARGS)
        assert LimitState.TENSION_FLANGE_YIELDING not in {s.limit_state for s in states}

    def test_tension_flange_yielding_applies_when_Sxt_is_smaller(
        self, girder: SectionAdapter
    ) -> None:
        kwargs = {**self.KWARGS, "Sxt": 260.0}
        states = F.f4_strength(F.FlexuralMember(girder, A992, Lb=0.0), **kwargs)
        tfy = next(s for s in states if s.limit_state is LimitState.TENSION_FLANGE_YIELDING)
        assert tfy.citation.equation == "F4-15"

    def test_low_Iyc_forces_J_to_zero_and_Rpc_to_unity(self, girder: SectionAdapter) -> None:
        """Sect. F4.2(2) and Eq. F4-10 share the Iyc/Iy <= 0.23 threshold.

        Dropping J shortens Lr and cuts the elastic LTB strength, so the member
        with the smaller compression flange must be weaker at a long Lb.
        """
        strong = F.f4_strength(F.FlexuralMember(girder, A992, Lb=400.0), **self.KWARGS)
        weak = F.f4_strength(
            F.FlexuralMember(girder, A992, Lb=400.0), **{**self.KWARGS, "Iyc_over_Iy": 0.20}
        )
        strong_ltb = next(
            s.nominal for s in strong if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING
        )
        weak_ltb = next(
            s.nominal for s in weak if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING
        )
        assert weak_ltb < strong_ltb


# ===========================================================================
# Sect. F5
# ===========================================================================
class TestF5:
    def test_Rpg_is_unity_at_the_5_7_threshold(self) -> None:
        """Eq. F5-6's bracket vanishes at hc/tw = 5.7*sqrt(E/Fy) -- Rpg = 1 exactly."""
        hc_tw = 5.7 * math.sqrt(E_STEEL / FY50)
        assert F.Rpg_factor(2.0, hc_tw, 1.0, E_STEEL, FY50) == pytest.approx(1.0, rel=1e-12)

    def test_Rpg_reduces_a_slender_web(self) -> None:
        assert F.Rpg_factor(5.0, 200.0, 1.0, E_STEEL, FY50) < 1.0

    def test_Rpg_never_exceeds_unity(self) -> None:
        assert F.Rpg_factor(5.0, 50.0, 1.0, E_STEEL, FY50) == 1.0

    def test_aw_is_capped_at_10_in_Rpg(self) -> None:
        """Sect. F5.2: aw "shall not exceed 10" -- but only inside Eq. F5-6."""
        assert F.Rpg_factor(10.0, 200.0, 1.0, E_STEEL, FY50) == pytest.approx(
            F.Rpg_factor(50.0, 200.0, 1.0, E_STEEL, FY50), rel=1e-12
        )
        # Eq. F4-11 applies no cap, so rt keeps falling past aw = 10.
        assert F.rt_effective_radius(8.0, 50.0) < F.rt_effective_radius(8.0, 10.0)

    def test_Lr_F5_has_no_torsional_term(self) -> None:
        """Eq. F5-5 is pi*rt*sqrt(E/(0.7*Fy)) -- no J, unlike Eq. F4-8."""
        assert F.Lr_F5(2.0, E_STEEL, FY50) == pytest.approx(
            math.pi * 2.0 * math.sqrt(E_STEEL / 35.0), rel=1e-12
        )

    def test_slender_web_girder_states(self) -> None:
        girder = SectionAdapter.from_mapping(
            {"name": "PG60", "Ag": 45.0, "Zx": 520.0, "Sx": 450.0, "ry": 2.1, "J": 3.0}
        )
        states = F.f5_strength(
            F.FlexuralMember(girder, A992, Lb=240.0), Sxc=450.0, Sxt=450.0, hc=58.0,
            bfc=14.0, tfc=0.75, tw=0.3125, flange_class=COMPACT,
            lam_f=9.33, lam_pf=9.15, lam_rf=24.08, lam_w=185.6,
        )
        kinds = {s.limit_state for s in states}
        assert LimitState.COMPRESSION_FLANGE_YIELDING in kinds
        assert LimitState.LATERAL_TORSIONAL_BUCKLING in kinds

    def test_F5_10_omits_Rpg(self) -> None:
        """Eq. F5-10 is Fy*Sxt with no Rpg -- the tension flange is not reduced."""
        girder = SectionAdapter.from_mapping(
            {"name": "PG60", "Ag": 45.0, "Zx": 520.0, "Sx": 450.0, "ry": 2.1, "J": 3.0}
        )
        states = F.f5_strength(
            F.FlexuralMember(girder, A992, Lb=0.0), Sxc=450.0, Sxt=400.0, hc=58.0,
            bfc=14.0, tfc=0.75, tw=0.3125, flange_class=COMPACT,
            lam_f=9.33, lam_pf=9.15, lam_rf=24.08, lam_w=185.6,
        )
        tfy = next(s for s in states if s.limit_state is LimitState.TENSION_FLANGE_YIELDING)
        assert tfy.nominal == pytest.approx(FY50 * 400.0, rel=1e-12)


# ===========================================================================
# Sect. F6
# ===========================================================================
class TestF6:
    def test_minor_axis_Mp_is_capped_at_1_6_Fy_Sy(self, w18x50: SectionAdapter) -> None:
        """W18x50: Zy = 19.0, Sy = 10.7 -> Fy*Zy = 950 but 1.6*Fy*Sy = 856 governs.

        The shape factor of a weak-axis I-shape is about 1.8, well past the cap.
        """
        states = F.f6_strength(
            compact_member(w18x50, axis=Axis.MINOR), flange_class=COMPACT,
            lam=6.57, lam_pf=9.15, lam_rf=24.08,
        )
        assert states[0].nominal == pytest.approx(1.6 * FY50 * 10.7, rel=1e-12)
        assert states[0].nominal < FY50 * 19.0

    def test_no_ltb_limit_state(self, w18x50: SectionAdapter) -> None:
        """Sect. F6 lists only yielding and FLB -- a weak-axis member cannot buckle laterally."""
        states = F.f6_strength(
            compact_member(w18x50, axis=Axis.MINOR, Lb=600.0), flange_class=COMPACT,
            lam=6.57, lam_pf=9.15, lam_rf=24.08,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING not in {s.limit_state for s in states}

    def test_slender_flange_uses_0_69E(self, w18x50: SectionAdapter) -> None:
        """Eq. F6-4: Fcr = 0.69E/lambda^2 -- note 0.69, not the 0.9*E*kc of Eq. F3-2."""
        states = F.f6_strength(
            compact_member(w18x50, axis=Axis.MINOR), flange_class=SLENDER,
            lam=30.0, lam_pf=9.15, lam_rf=24.08,
        )
        flb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert flb.nominal == pytest.approx(0.69 * E_STEEL / 900.0 * 10.7, rel=1e-12)


# ===========================================================================
# Sect. F7
# ===========================================================================
class TestF7:
    @pytest.fixture
    def hss(self) -> SectionAdapter:
        return SectionAdapter.from_mapping(
            {"name": "HSS10X6X1/2", "Ag": 13.5, "Ix": 169.0, "Iy": 76.8, "Zx": 40.9,
             "Sx": 33.8, "Zy": 28.2, "Sy": 25.6, "ry": 2.39, "rx": 3.54, "J": 158.0,
             "Ht": 10.0, "B": 6.0}
        )

    def test_yielding(self, hss: SectionAdapter) -> None:
        states = F.f7_strength(
            F.FlexuralMember(hss, A500_C_RECT, shape=F.ShapeType.RECTANGULAR_HSS, Lb=0.0),
            flange_class=COMPACT, web_class=COMPACT, b_over_t=10.0, h_over_tw=18.0,
            b=4.65, t=0.465,
        )
        assert states[0].nominal == pytest.approx(50.0 * 40.9, rel=1e-12)

    def test_F7_2_interpolates_on_b_over_t_not_lambda(self, hss: SectionAdapter) -> None:
        """Eq. F7-2 uses (3.57*(b/t)*sqrt(Fy/E) - 4.0) directly, not (lam-lam_p)/(lam_r-lam_p).

        At b/t = lambda_p = 1.12*sqrt(E/Fy) the bracket is 3.57*1.12 - 4.0 = -0.0016,
        i.e. essentially zero -- so Eq. F7-2 returns Mp there, but only because
        3.57*1.12 happens to be 3.9984. The near-miss is the Specification's, not a bug.
        """
        member = F.FlexuralMember(
            hss, A500_C_RECT, shape=F.ShapeType.RECTANGULAR_HSS, Lb=0.0
        )
        lam_p, _ = flexural_limits(FlexuralElement.RECTANGULAR_HSS_FLANGE, E_STEEL, 50.0)
        states = F.f7_strength(
            member, flange_class=NONCOMPACT, web_class=COMPACT,
            b_over_t=lam_p, h_over_tw=18.0, b=4.65, t=0.465,
        )
        flb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert flb.nominal == pytest.approx(50.0 * 40.9, rel=1e-3)
        assert pytest.approx(4.0, abs=0.002) == 3.57 * 1.12

    def test_effective_width_hss_vs_box_coefficient(self) -> None:
        """Eqs. F7-4 and F7-5 differ only in 0.38 vs 0.34; the box value is larger."""
        hss_be = F.hss_effective_flange_width(10.0, 0.25, E_STEEL, 50.0, box=False)
        box_be = F.hss_effective_flange_width(10.0, 0.25, E_STEEL, 50.0, box=True)
        assert box_be > hss_be

    def test_effective_width_is_capped_at_b_near_lambda_r(self) -> None:
        """Just above lambda_r = 33.7 the formula returns ~b, and the cap binds."""
        t = 0.25
        b = 33.72 * t
        assert F.hss_effective_flange_width(b, t, E_STEEL, 50.0) == pytest.approx(b, rel=1e-3)

    def test_a_stocky_flange_is_out_of_scope_for_F7_4(self) -> None:
        """Below b/t = 0.38*sqrt(E/Fy) = 9.2 the bracket goes negative.

        Returning zero effective width there would silently zero the strength of
        a perfectly stocky flange, so it raises instead.
        """
        with pytest.raises(OutOfScopeError, match="not slender"):
            F.hss_effective_flange_width(2.0, 0.5, E_STEEL, 50.0)

    def test_effective_section_modulus_is_below_gross(self, hss: SectionAdapter) -> None:
        Se = F.effective_section_modulus_rect_hss(13.5, 169.0, 10.0, 0.465, 4.65, 3.5)
        assert 0.0 < Se < 33.8

    def test_effective_section_modulus_rejects_a_non_slender_flange(self) -> None:
        with pytest.raises(GeometryError, match="not less than"):
            F.effective_section_modulus_rect_hss(13.5, 169.0, 10.0, 0.465, 4.65, 4.65)

    def test_square_section_has_no_ltb(self) -> None:
        """User Note, p. 16.1-59: LTB cannot occur in a square section."""
        square = SectionAdapter.from_mapping(
            {"name": "HSS8X8X1/2", "Ag": 13.5, "Ix": 125.0, "Iy": 125.0, "Zx": 37.5,
             "Sx": 31.2, "Zy": 37.5, "Sy": 31.2, "ry": 3.04, "J": 196.0, "Ht": 8.0}
        )
        states = F.f7_strength(
            F.FlexuralMember(square, A500_C_RECT, shape=F.ShapeType.RECTANGULAR_HSS, Lb=600.0),
            flange_class=COMPACT, web_class=COMPACT, b_over_t=14.2, h_over_tw=14.2,
            b=6.6, t=0.465,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING not in {s.limit_state for s in states}

    def test_minor_axis_has_no_ltb(self, hss: SectionAdapter) -> None:
        states = F.f7_strength(
            F.FlexuralMember(hss, A500_C_RECT, shape=F.ShapeType.RECTANGULAR_HSS,
                             Lb=600.0, axis=Axis.MINOR),
            flange_class=COMPACT, web_class=COMPACT, b_over_t=10.0, h_over_tw=18.0,
            b=4.65, t=0.465,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING not in {s.limit_state for s in states}

    def test_rectangular_section_does_get_ltb(self, hss: SectionAdapter) -> None:
        states = F.f7_strength(
            F.FlexuralMember(hss, A500_C_RECT, shape=F.ShapeType.RECTANGULAR_HSS, Lb=900.0),
            flange_class=COMPACT, web_class=COMPACT, b_over_t=10.0, h_over_tw=18.0,
            b=4.65, t=0.465,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING in {s.limit_state for s in states}

    def test_slender_hss_web_is_rejected(self, hss: SectionAdapter) -> None:
        """User Note, p. 16.1-58: "There are no HSS with slender webs"."""
        with pytest.raises(OutOfScopeError, match="no HSS with slender webs"):
            F.f7_strength(
                F.FlexuralMember(hss, A500_C_RECT, shape=F.ShapeType.RECTANGULAR_HSS, Lb=0.0),
                flange_class=COMPACT, web_class=SLENDER, b_over_t=10.0, h_over_tw=200.0,
                b=4.65, t=0.465,
            )


# ===========================================================================
# Sect. F8
# ===========================================================================
class TestF8:
    @pytest.fixture
    def pipe(self) -> SectionAdapter:
        return SectionAdapter.from_mapping(
            {"name": "HSS10.000X0.375", "Ag": 10.6, "Zx": 34.0, "Sx": 25.9, "rx": 3.42}
        )

    def test_compact_wall_yields(self, pipe: SectionAdapter) -> None:
        states = F.f8_strength(
            F.FlexuralMember(pipe, A500_C_RECT, shape=F.ShapeType.ROUND_HSS),
            D_over_t=28.7, wall_class=COMPACT,
        )
        assert len(states) == 1
        assert states[0].nominal == pytest.approx(50.0 * 34.0, rel=1e-12)

    def test_noncompact_wall(self, pipe: SectionAdapter) -> None:
        states = F.f8_strength(
            F.FlexuralMember(pipe, A500_C_RECT, shape=F.ShapeType.ROUND_HSS),
            D_over_t=60.0, wall_class=NONCOMPACT,
        )
        lb = next(s for s in states if s.limit_state is LimitState.LOCAL_BUCKLING)
        assert lb.nominal == pytest.approx((0.021 * E_STEEL / 60.0 + 50.0) * 25.9, rel=1e-12)

    def test_slender_wall(self, pipe: SectionAdapter) -> None:
        states = F.f8_strength(
            F.FlexuralMember(pipe, A500_C_RECT, shape=F.ShapeType.ROUND_HSS),
            D_over_t=200.0, wall_class=SLENDER,
        )
        lb = next(s for s in states if s.limit_state is LimitState.LOCAL_BUCKLING)
        assert lb.nominal == pytest.approx(0.33 * E_STEEL / 200.0 * 25.9, rel=1e-12)

    def test_beyond_0_45_E_over_Fy_is_out_of_scope(self, pipe: SectionAdapter) -> None:
        """Sect. F8's opening sentence limits D/t to less than 0.45E/Fy = 261."""
        with pytest.raises(OutOfScopeError, match="0.45"):
            F.f8_strength(
                F.FlexuralMember(pipe, A500_C_RECT, shape=F.ShapeType.ROUND_HSS),
                D_over_t=300.0, wall_class=SLENDER,
            )


# ===========================================================================
# Sect. F9
# ===========================================================================
class TestF9:
    @pytest.fixture
    def tee(self) -> SectionAdapter:
        return SectionAdapter.from_mapping(
            {"name": "WT9X25", "Ag": 7.33, "Zx": 11.2, "Sx": 6.05, "Iy": 22.5,
             "J": 0.435, "ry": 1.75, "rx": 2.65, "d": 9.0}
        )

    def test_stem_in_tension_uses_F9_2_with_the_1_6_cap(self, tee: SectionAdapter) -> None:
        states = F.f9_strength(
            F.FlexuralMember(tee, A992, shape=F.ShapeType.TEE, Lb=0.0, stem_in_compression=False),
            d=9.0, Sxc=6.05, flange_class=COMPACT, lam_f=6.0, lam_pf=9.15, lam_rf=24.08,
            d_over_tw=20.0,
        )
        My = FY50 * 6.05
        assert states[0].citation.equation == "F9-2"
        assert states[0].nominal == pytest.approx(min(FY50 * 11.2, 1.6 * My), rel=1e-12)

    def test_stem_in_compression_caps_Mp_at_My(self, tee: SectionAdapter) -> None:
        """Eq. F9-4: for tee stems in compression, Mp = My -- no shape factor at all."""
        states = F.f9_strength(
            F.FlexuralMember(tee, A992, shape=F.ShapeType.TEE, Lb=0.0, stem_in_compression=True),
            d=9.0, Sxc=6.05, flange_class=COMPACT, lam_f=6.0, lam_pf=9.15, lam_rf=24.08,
            d_over_tw=20.0,
        )
        assert states[0].citation.equation == "F9-4"
        assert states[0].nominal == pytest.approx(FY50 * 6.05, rel=1e-12)

    def test_double_angle_web_legs_in_compression_use_1_5_My(self, tee: SectionAdapter) -> None:
        """Eq. F9-5 gives 1.5*My, unlike the tee's My of Eq. F9-4."""
        states = F.f9_strength(
            F.FlexuralMember(tee, A992, shape=F.ShapeType.DOUBLE_ANGLE, Lb=0.0,
                             stem_in_compression=True),
            d=9.0, Sxc=6.05, flange_class=COMPACT, lam_f=6.0, lam_pf=9.15, lam_rf=24.08,
            d_over_tw=20.0,
        )
        assert states[0].citation.equation == "F9-5"
        assert states[0].nominal == pytest.approx(1.5 * FY50 * 6.05, rel=1e-12)

    def test_the_sign_of_B_dominates_the_ltb_strength(self) -> None:
        """Eqs. F9-11 vs F9-12: the same tee is far weaker with its stem in compression.

        B + sqrt(1+B^2) collapses towards zero as B goes negative, so getting
        the sign wrong is unconservative by a large factor.
        """
        tension = F.Mcr_F9(1.0, 120.0, E_STEEL, 22.5, 0.435, 9.0, stem_in_compression=False)
        compression = F.Mcr_F9(1.0, 120.0, E_STEEL, 22.5, 0.435, 9.0, stem_in_compression=True)
        assert compression < tension
        assert tension / compression > 5.0

    def test_Mcr_reduces_to_the_pure_torsion_value_at_B_zero(self) -> None:
        """As Lb grows, B -> 0 and Eq. F9-10 tends to 1.95*E*sqrt(Iy*J)/Lb."""
        Lb = 1.0e6
        got = F.Mcr_F9(1.0, Lb, E_STEEL, 22.5, 0.435, 9.0, stem_in_compression=False)
        assert got == pytest.approx(1.95 * E_STEEL / Lb * math.sqrt(22.5 * 0.435), rel=1e-3)

    def test_tee_stem_local_buckling_branches(self, tee: SectionAdapter) -> None:
        root = math.sqrt(E_STEEL / FY50)
        for d_tw, expected_eq in (
            (0.5 * root, "F9-17"), (1.0 * root, "F9-18"), (2.0 * root, "F9-19")
        ):
            states = F.f9_strength(
                F.FlexuralMember(tee, A992, shape=F.ShapeType.TEE, Lb=0.0,
                                 stem_in_compression=True),
                d=9.0, Sxc=6.05, flange_class=COMPACT, lam_f=6.0, lam_pf=9.15,
                lam_rf=24.08, d_over_tw=d_tw,
            )
            wlb = next(s for s in states if s.limit_state is LimitState.WEB_LOCAL_BUCKLING)
            assert expected_eq in wlb.note

    def test_F9_18_is_continuous_with_F9_17_at_0_84(self, tee: SectionAdapter) -> None:
        """At d/tw = 0.84*sqrt(E/Fy), Eq. F9-18 gives (1.43 - 0.515*0.84)*Fy = 0.9974*Fy.

        A 0.26% step, not exact -- the Specification's own rounding of 1.43.
        """
        boundary = 1.43 - 0.515 * 0.84
        assert boundary == pytest.approx(0.9974, abs=1e-4)
        assert boundary < 1.0


# ===========================================================================
# Sect. F10
# ===========================================================================
class TestF10:
    def test_yielding_is_1_5_My(self) -> None:
        angle = SectionAdapter.from_mapping({"name": "L4X4X1/2", "Ag": 3.75})
        states = F.f10_strength(
            F.FlexuralMember(angle, A36, shape=F.ShapeType.SINGLE_ANGLE),
            My=36.0 * 1.05, Sc=1.05, b_over_t=8.0, leg_class=COMPACT,
        )
        assert states[0].nominal == pytest.approx(1.5 * 36.0 * 1.05, rel=1e-12)

    def test_cb_is_capped_at_1_5(self) -> None:
        """Sect. F10.2: "Cb is computed using Equation F1-1 with a maximum value of 1.5".

        The only Cb cap anywhere in Chapter F.
        """
        at_cap = F.Mcr_F10_principal(1.5, E_STEEL, 3.75, 0.778, 0.5, 120.0, 0.0)
        above = F.Mcr_F10_principal(2.5, E_STEEL, 3.75, 0.778, 0.5, 120.0, 0.0)
        assert above == pytest.approx(at_cap, rel=1e-12)

    def test_beta_w_sign_changes_the_strength(self) -> None:
        """beta_w is negative when the long leg is in compression -- the weaker case."""
        short_leg_compression = F.Mcr_F10_principal(1.0, E_STEEL, 4.75, 0.87, 0.5, 120.0, 3.0)
        long_leg_compression = F.Mcr_F10_principal(1.0, E_STEEL, 4.75, 0.87, 0.5, 120.0, -3.0)
        assert long_leg_compression < short_leg_compression

    def test_equal_leg_angle_has_zero_beta_w(self) -> None:
        """With beta_w = 0 the bracket reduces to sqrt(1) + 0 = 1."""
        got = F.Mcr_F10_principal(1.0, E_STEEL, 3.75, 0.778, 0.5, 120.0, 0.0)
        assert got == pytest.approx(
            9.0 * E_STEEL * 3.75 * 0.778 * 0.5 / (8.0 * 120.0), rel=1e-12
        )

    def test_geometric_axis_toe_in_compression_is_weaker(self) -> None:
        """Eq. F10-5a (minus one) against Eq. F10-5b (plus one)."""
        compression = F.Mcr_F10_geometric(
            1.0, E_STEEL, 4.0, 0.5, 120.0, compression_at_toe=True
        )
        tension = F.Mcr_F10_geometric(
            1.0, E_STEEL, 4.0, 0.5, 120.0, compression_at_toe=False
        )
        assert compression < tension

    def test_restraint_at_max_moment_gives_the_1_25_factor(self) -> None:
        """Sect. F10.2(2)(ii)."""
        plain = F.Mcr_F10_geometric(1.0, E_STEEL, 4.0, 0.5, 120.0, compression_at_toe=True)
        restrained = F.Mcr_F10_geometric(
            1.0, E_STEEL, 4.0, 0.5, 120.0, compression_at_toe=True,
            restrained_at_max_moment=True,
        )
        assert restrained == pytest.approx(1.25 * plain, rel=1e-12)

    def test_ltb_branch_selection(self) -> None:
        angle = SectionAdapter.from_mapping({"name": "L4X4X1/2", "Ag": 3.75})
        member = F.FlexuralMember(angle, A36, shape=F.ShapeType.SINGLE_ANGLE, Lb=120.0)
        My = 36.0 * 1.05
        stocky = F.f10_strength(member, My=My, Sc=1.05, b_over_t=8.0,
                                leg_class=COMPACT, Mcr=My * 2.0)
        slender = F.f10_strength(member, My=My, Sc=1.05, b_over_t=8.0,
                                 leg_class=COMPACT, Mcr=My * 0.5)
        assert next(
            s.citation.equation for s in stocky
            if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING
        ) == "F10-2"
        assert next(
            s.citation.equation for s in slender
            if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING
        ) == "F10-3"

    def test_F10_2_is_capped_at_1_5_My(self) -> None:
        angle = SectionAdapter.from_mapping({"name": "L4X4X1/2", "Ag": 3.75})
        My = 36.0 * 1.05
        states = F.f10_strength(
            F.FlexuralMember(angle, A36, shape=F.ShapeType.SINGLE_ANGLE, Lb=12.0),
            My=My, Sc=1.05, b_over_t=8.0, leg_class=COMPACT, Mcr=My * 1000.0,
        )
        ltb = next(s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING)
        assert ltb.nominal == pytest.approx(1.5 * My, rel=1e-12)

    def test_leg_local_buckling_only_when_the_toe_is_in_compression(self) -> None:
        """Sect. F10.3: "applies when the toe of the leg is in compression"."""
        angle = SectionAdapter.from_mapping({"name": "L4X4X1/4", "Ag": 1.93})
        member = F.FlexuralMember(angle, A36, shape=F.ShapeType.SINGLE_ANGLE)
        with_toe = F.f10_strength(member, My=36.0 * 0.56, Sc=0.56, b_over_t=16.0,
                                  leg_class=SLENDER, toe_in_compression=True)
        without = F.f10_strength(member, My=36.0 * 0.56, Sc=0.56, b_over_t=16.0,
                                 leg_class=SLENDER, toe_in_compression=False)
        assert LimitState.LEG_LOCAL_BUCKLING in {s.limit_state for s in with_toe}
        assert LimitState.LEG_LOCAL_BUCKLING not in {s.limit_state for s in without}


# ===========================================================================
# Sect. F11
# ===========================================================================
class TestF11:
    @pytest.fixture
    def bar(self) -> SectionAdapter:
        return SectionAdapter.from_mapping({"name": "BAR4X1", "Ag": 4.0})

    def test_Mp_is_capped_at_1_6_Fy_S(self, bar: SectionAdapter) -> None:
        """A solid rectangle's shape factor is 1.5, comfortably under the cap."""
        Z, S = 1.0 * 4.0**2 / 4.0, 1.0 * 4.0**2 / 6.0
        states = F.f11_strength(
            F.FlexuralMember(bar, A36, shape=F.ShapeType.RECTANGULAR_BAR, Lb=0.0),
            d=4.0, t=1.0, Z=Z, S=S,
        )
        assert states[0].nominal == pytest.approx(36.0 * Z, rel=1e-12)
        assert 36.0 * Z < 1.6 * 36.0 * S

    def test_no_ltb_below_0_08_E_over_Fy(self, bar: SectionAdapter) -> None:
        """Sect. F11.2(a). At Fy = 36, 0.08E/Fy = 64.4; Lb*d/t^2 = 40 is below it."""
        states = F.f11_strength(
            F.FlexuralMember(bar, A36, shape=F.ShapeType.RECTANGULAR_BAR, Lb=10.0),
            d=4.0, t=1.0, Z=4.0, S=2.667,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING not in {s.limit_state for s in states}

    def test_F11_2_in_the_middle_band(self, bar: SectionAdapter) -> None:
        states = F.f11_strength(
            F.FlexuralMember(bar, A36, shape=F.ShapeType.RECTANGULAR_BAR, Lb=60.0),
            d=4.0, t=1.0, Z=4.0, S=2.667,
        )
        ltb = next(s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING)
        assert ltb.citation.equation == "F11-2"

    def test_F11_3_beyond_1_9_E_over_Fy(self, bar: SectionAdapter) -> None:
        states = F.f11_strength(
            F.FlexuralMember(bar, A36, shape=F.ShapeType.RECTANGULAR_BAR, Lb=600.0),
            d=4.0, t=1.0, Z=4.0, S=2.667,
        )
        ltb = next(s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING)
        assert ltb.citation.equation == "F11-3"

    def test_rounds_have_no_ltb(self, bar: SectionAdapter) -> None:
        """Sect. F11.2(d): LTB need not be considered for rounds."""
        states = F.f11_strength(
            F.FlexuralMember(bar, A36, shape=F.ShapeType.ROUND_BAR, Lb=600.0),
            d=4.0, t=4.0, Z=4.0, S=2.667,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING not in {s.limit_state for s in states}

    def test_minor_axis_bars_have_no_ltb(self, bar: SectionAdapter) -> None:
        states = F.f11_strength(
            F.FlexuralMember(bar, A36, shape=F.ShapeType.RECTANGULAR_BAR, Lb=600.0,
                             axis=Axis.MINOR),
            d=4.0, t=1.0, Z=4.0, S=2.667,
        )
        assert LimitState.LATERAL_TORSIONAL_BUCKLING not in {s.limit_state for s in states}


# ===========================================================================
# Sect. F12 and F13
# ===========================================================================
class TestF12:
    def test_yielding_only_is_flagged_incomplete(self) -> None:
        """Sect. F12 needs Fcr from analysis; without it the check is partial and says so."""
        z = SectionAdapter.from_mapping({"name": "Z-section", "Ag": 5.0})
        states = F.f12_strength(
            F.FlexuralMember(z, A992, shape=F.ShapeType.UNSYMMETRICAL), S_min=20.0
        )
        assert "INCOMPLETE" in states[0].note

    def test_supplied_stresses_are_capped_at_Fy(self) -> None:
        z = SectionAdapter.from_mapping({"name": "Z-section", "Ag": 5.0})
        states = F.f12_strength(
            F.FlexuralMember(z, A992, shape=F.ShapeType.UNSYMMETRICAL),
            S_min=20.0, Fcr_ltb=80.0, Fcr_local=30.0,
        )
        ltb = next(s for s in states if s.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING)
        assert ltb.nominal == pytest.approx(FY50 * 20.0, rel=1e-12)  # capped at Fy
        assert "INCOMPLETE" not in states[0].note


class TestF13:
    def test_rupture_does_not_apply_when_the_net_flange_is_strong_enough(self) -> None:
        assert F.tension_rupture_cap(65.0, 5.0, 5.5, 50.0, 88.9) is None

    def test_Yt_switches_at_Fy_over_Fu_of_0_8(self) -> None:
        """Yt = 1.0 for Fy/Fu <= 0.8, else 1.1. A992 is 50/65 = 0.769 -> Yt = 1.0."""
        assert 50.0 / 65.0 < 0.8
        borderline = F.tension_rupture_cap(60.0, 3.5, 5.5, 50.0, 88.9)
        assert borderline is not None

    def test_rupture_cap_value(self) -> None:
        cap = F.tension_rupture_cap(65.0, 3.0, 5.5, 50.0, 88.9)
        assert cap == pytest.approx(65.0 * 3.0 / 5.5 * 88.9, rel=1e-12)

    def test_net_area_above_gross_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="exceeds gross"):
            F.tension_rupture_cap(65.0, 6.0, 5.5, 50.0, 88.9)

    @pytest.mark.parametrize("ratio", [0.05, 0.95])
    def test_proportioning_limit_rejects_extremes(self, ratio: float) -> None:
        """Eq. F13-2: 0.1 <= Iyc/Iy <= 0.9."""
        with pytest.raises(OutOfScopeError, match="F13-2"):
            F.check_proportioning_limits(ratio * 100.0, 100.0)

    @pytest.mark.parametrize("ratio", [0.1, 0.5, 0.9])
    def test_proportioning_limit_accepts_the_band(self, ratio: float) -> None:
        F.check_proportioning_limits(ratio * 100.0, 100.0)

    def test_web_slenderness_limits(self) -> None:
        """Eqs. F13-3 and F13-4, and the 260 cap for an unstiffened girder."""
        assert F.web_slenderness_limit(E_STEEL, FY50, 1.0) == pytest.approx(
            12.0 * math.sqrt(E_STEEL / FY50), rel=1e-12
        )
        assert F.web_slenderness_limit(E_STEEL, FY50, 2.0) == pytest.approx(
            0.40 * E_STEEL / FY50, rel=1e-12
        )
        assert F.web_slenderness_limit(E_STEEL, FY50, None) == 260.0


# ===========================================================================
# Orchestrator
# ===========================================================================
class TestDispatch:
    @pytest.mark.parametrize(
        ("flange", "web", "expected"),
        [
            (COMPACT, COMPACT, "F2"),
            (NONCOMPACT, COMPACT, "F3"),
            (SLENDER, COMPACT, "F3"),
            (COMPACT, NONCOMPACT, "F4"),
            (NONCOMPACT, NONCOMPACT, "F4"),
            (COMPACT, SLENDER, "F5"),
            (SLENDER, SLENDER, "F5"),
        ],
    )
    def test_section_selection(
        self, flange: FlexuralSlenderness, web: FlexuralSlenderness, expected: str
    ) -> None:
        assert F.selected_section(flange, web) == expected

    def test_lrfd_and_asd_share_one_nominal_strength(self, w18x50: SectionAdapter) -> None:
        member = compact_member(w18x50, Lb=120.0)
        lrfd = F.flexural_strength(member, flange_class=COMPACT, web_class=COMPACT)
        asd = F.flexural_strength(
            member, basis=Basis.ASD, flange_class=COMPACT, web_class=COMPACT
        )
        assert lrfd.nominal == pytest.approx(asd.nominal, rel=1e-12)
        assert lrfd.available / asd.available == pytest.approx(0.90 * 1.67, rel=1e-12)

    def test_phi_and_omega_are_the_chapter_f_values(self, w18x50: SectionAdapter) -> None:
        """Sect. F1(a), p. 16.1-46: phi_b = 0.90, Omega_b = 1.67 for the whole chapter."""
        result = F.flexural_strength(
            compact_member(w18x50), flange_class=COMPACT, web_class=COMPACT
        )
        assert (result.phi, result.omega) == (0.90, 1.67)

    def test_minor_axis_routes_to_F6(self, w18x50: SectionAdapter) -> None:
        result = F.flexural_strength(
            compact_member(w18x50, axis=Axis.MINOR),
            flange_class=COMPACT, lam=6.57, lam_pf=9.15, lam_rf=24.08,
        )
        assert result.citation.equation == "F6-1"

    def test_report_renders(self, w18x50: SectionAdapter) -> None:
        result = F.flexural_strength(
            compact_member(w18x50, Lb=200.0), flange_class=COMPACT, web_class=COMPACT
        )
        text = result.report(required=3000.0)
        assert "lateral-torsional buckling" in text
        assert "governs" in text
        assert "OK" in text


# ===========================================================================
# Published benchmarks
# ===========================================================================
_EXAMPLES = Path(__file__).parent / "design_examples" / "chapter_f.json"


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
    return [case for case in data.get("cases", []) if not case.get("skip")]


class TestPublishedBenchmarks:
    """AISC *Manual* Table 3-2, at the table's own three-significant-figure precision."""

    @pytest.mark.design_example
    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: str(c.get("id", "?")))
    def test_matches_the_published_values(self, case: dict[str, Any]) -> None:
        from pyaisc360.materials import grade

        section = SectionAdapter.from_mapping(case["section"])
        steel = grade(str(case["grade"]))
        tol = float(case.get("tol", 0.005))
        props = case["section"]

        if case.get("Lp_ft") is not None:
            Lp = F.Lp_F2(props["ry"], steel.E, steel.Fy) / 12.0
            assert Lp == pytest.approx(case["Lp_ft"], rel=tol)

        if case.get("Lr_ft") is not None:
            Lr = F.Lr_F2(
                props["rts"], steel.E, steel.Fy, props["J"], 1.0, props["Sx"], props["ho"]
            ) / 12.0
            assert Lr == pytest.approx(case["Lr_ft"], rel=tol)

        if case.get("phiMn_kipft") is not None:
            member = F.FlexuralMember(
                section, steel, shape=F.ShapeType.ROLLED_I,
                Lb=float(case["Lb"]), Cb=float(case["Cb"]),
            )
            result = F.flexural_strength(member, flange_class=COMPACT, web_class=COMPACT)
            assert result.available / 12.0 == pytest.approx(case["phiMn_kipft"], rel=tol)

    def test_every_case_declares_its_source(self) -> None:
        """The provenance rule: no expected value without a stated source."""
        for case in _load_cases():
            assert case.get("source"), f"case {case.get('id')!r} has no source"

    def test_all_four_benchmark_shapes_are_present(self) -> None:
        ids = " ".join(str(c["id"]) for c in _load_cases())
        for name in ("W18X50", "W16X26", "W24X68", "W24X55"):
            assert name in ids

    def test_benchmark_shapes_are_compact_at_fy_50(self) -> None:
        """Every Table 3-2 row above is a Sect. F2 member -- confirmed, not assumed."""
        lam_pf, _ = flexural_limits(FlexuralElement.ROLLED_I_FLANGE, E_STEEL, FY50)
        lam_pw, _ = flexural_limits(FlexuralElement.I_WEB, E_STEEL, FY50)
        for name, props in SHAPES.items():
            assert props["bf_2tf"] <= lam_pf, f"{name} flange"
            assert props["h_tw"] <= lam_pw, f"{name} web"


class TestF13CoverPlatesAndRedistribution:
    @pytest.mark.parametrize(
        ("end_weld", "factor", "equation"),
        [("thick", 1.0, "F13-5"), ("thin", 1.5, "F13-6"), ("none", 2.0, "F13-7")],
    )
    def test_cover_plate_extension(self, end_weld: str, factor: float, equation: str) -> None:
        """Eqs. F13-5/6/7, Sect. F13.3(e): a' = w, 1.5w or 2w by end-weld detail."""
        assert F.cover_plate_extension(8.0, end_weld=end_weld) == pytest.approx(
            factor * 8.0, rel=1e-12
        )

    def test_cover_plate_extension_rejects_an_unknown_detail(self) -> None:
        with pytest.raises(Exception, match="end_weld"):
            F.cover_plate_extension(8.0, end_weld="tack")

    def test_Lm_reverse_curvature_is_longer_than_single(self) -> None:
        """Eq. F13-8. M1/M2 is POSITIVE for reverse curvature -- the same
        convention Eq. A-8-4 uses for Cm (p. 16.1-250)."""
        reverse = F.Lm_moment_redistribution(1.0, 1.65, E_STEEL, FY50)
        single = F.Lm_moment_redistribution(-1.0, 1.65, E_STEEL, FY50)
        assert reverse > single
        assert reverse == pytest.approx(
            (0.12 + 0.076) * (E_STEEL / FY50) * 1.65, rel=1e-12
        )

    def test_Lm_box_has_a_floor(self) -> None:
        """Eq. F13-9 floors at 0.10*(E/Fy)*ry; at M1/M2 = -1 the bracket is 0.07."""
        got = F.Lm_moment_redistribution(-1.0, 1.65, E_STEEL, FY50, solid_or_box=True)
        assert got == pytest.approx(0.10 * (E_STEEL / FY50) * 1.65, rel=1e-12)

    def test_Lm_rejects_a_ratio_outside_the_unit_interval(self) -> None:
        with pytest.raises(GeometryError, match="is outside"):
            F.Lm_moment_redistribution(1.5, 1.65, E_STEEL, FY50)


class TestF7SlenderWebBox:
    def test_slender_web_box_evaluates_F7_7_and_F7_8(self) -> None:
        """Sect. F7.3(c) is reachable only for a built-up box, and yields both
        compression flange yielding (Eq. F7-7) and CFLB (Eqs. F7-8/F7-9)."""
        box = SectionAdapter.from_mapping(
            {"name": "BOX36X12", "Ag": 42.0, "Ix": 8200.0, "Iy": 1500.0, "Zx": 520.0,
             "Sx": 455.0, "ry": 5.98, "J": 3000.0, "Ht": 36.0}
        )
        states = F.f7_strength(
            F.FlexuralMember(box, A992, shape=F.ShapeType.BOX, Lb=0.0),
            flange_class=COMPACT, web_class=SLENDER,
            b_over_t=48.0, h_over_tw=140.0, b=12.0, t=0.25,
        )
        kinds = {s.limit_state for s in states}
        assert LimitState.COMPRESSION_FLANGE_YIELDING in kinds
        assert LimitState.FLANGE_LOCAL_BUCKLING in kinds
        cflb = next(s for s in states if s.limit_state is LimitState.FLANGE_LOCAL_BUCKLING)
        assert cflb.citation.equation == "F7-8"
        assert "kc = 4.0" in cflb.note
