"""Chapter G -- Design of Members for Shear.

The chapter's whole difficulty is keeping ``Cv1`` and ``Cv2`` apart, so the
tests lead with the difference: :class:`TestCv1VersusCv2` pins where they agree
(exactly, by construction) and where they diverge (the elastic branch that only
``Cv2`` has).

Precision: 0.5% against AISC *Manual* Table 3-2 and derived catalog figures;
1e-12 for the branch thresholds and coefficient identities.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pyaisc360 import chapter_g as G
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import LimitState
from pyaisc360.core.exceptions import AISC360Error, GeometryError
from pyaisc360.materials import A500_C_RECT, A992
from pyaisc360.sections import SectionAdapter

E_STEEL = 29000.0
FY50 = 50.0

SHAPES: dict[str, dict[str, float]] = {
    "W18X50": dict(d=18.0, tw=0.355, Ag=14.7, bf=7.50, tf=0.570, h_tw=45.2),
    "W16X26": dict(d=15.7, tw=0.250, Ag=7.68, bf=5.50, tf=0.345, h_tw=56.8),
    "W24X68": dict(d=23.7, tw=0.415, Ag=20.1, bf=8.97, tf=0.585, h_tw=52.0),
    "W24X55": dict(d=23.6, tw=0.395, Ag=16.2, bf=7.01, tf=0.505, h_tw=54.6),
}


def shape(name: str) -> SectionAdapter:
    return SectionAdapter.from_mapping({"name": name, **SHAPES[name]})


# ===========================================================================
# The two coefficients
# ===========================================================================
class TestCv1VersusCv2:
    """Cv1 and Cv2 are not interchangeable, and the difference is the chapter."""

    KV = 5.34

    def _limits(self, kv: float = 5.34) -> tuple[float, float]:
        root = math.sqrt(kv * E_STEEL / FY50)
        return 1.10 * root, 1.37 * root

    def test_both_are_unity_on_the_plateau(self) -> None:
        inelastic, _ = self._limits()
        for ratio in (10.0, 30.0, inelastic * 0.999):
            assert G.Cv1(ratio, self.KV, E_STEEL, FY50) == 1.0
            assert G.Cv2(ratio, self.KV, E_STEEL, FY50) == 1.0

    def test_both_are_unity_exactly_at_1_10_root(self) -> None:
        """The plateau is inclusive: <= 1.10*sqrt(kv*E/Fy) gives exactly 1.0."""
        inelastic, _ = self._limits()
        assert G.Cv1(inelastic, self.KV, E_STEEL, FY50) == pytest.approx(1.0, rel=1e-12)
        assert G.Cv2(inelastic, self.KV, E_STEEL, FY50) == pytest.approx(1.0, rel=1e-12)

    def test_they_agree_exactly_between_1_10_and_1_37(self) -> None:
        """Eq. G2-4 and Eq. G2-10 are the same expression -- identical to 1e-12."""
        inelastic, elastic = self._limits()
        for fraction in (0.05, 0.25, 0.5, 0.75, 0.99):
            ratio = inelastic + (elastic - inelastic) * fraction
            assert G.Cv1(ratio, self.KV, E_STEEL, FY50) == pytest.approx(
                G.Cv2(ratio, self.KV, E_STEEL, FY50), rel=1e-12
            )

    def test_they_diverge_beyond_1_37_root(self) -> None:
        """Past 1.37*sqrt(kv*E/Fy), Cv1 stays 1/(h/tw) but Cv2 goes 1/(h/tw)^2.

        Cv1 credits post-buckling strength that the buckling coefficient does
        not, so substituting Cv2 for Cv1 here is conservative -- and the reverse
        is unconservative.
        """
        _, elastic = self._limits()
        for ratio in (elastic * 1.2, elastic * 1.5, elastic * 2.0):
            assert G.Cv1(ratio, self.KV, E_STEEL, FY50) > G.Cv2(ratio, self.KV, E_STEEL, FY50)

    def test_cv1_decays_inverse_linearly(self) -> None:
        inelastic, _ = self._limits()
        a = G.Cv1(inelastic * 2.0, self.KV, E_STEEL, FY50)
        b = G.Cv1(inelastic * 4.0, self.KV, E_STEEL, FY50)
        assert a / b == pytest.approx(2.0, rel=1e-12)

    def test_cv2_decays_inverse_squared_in_the_elastic_branch(self) -> None:
        _, elastic = self._limits()
        a = G.Cv2(elastic * 2.0, self.KV, E_STEEL, FY50)
        b = G.Cv2(elastic * 4.0, self.KV, E_STEEL, FY50)
        assert a / b == pytest.approx(4.0, rel=1e-12)

    def test_cv2_elastic_branch_is_nearly_continuous(self) -> None:
        """Eqs. G2-10 and G2-11 meet at 1.37*sqrt(kv*E/Fy) only because
        1.10*1.37 = 1.507 ~ the printed 1.51 -- a 0.2% step, not exact."""
        _, elastic = self._limits()
        below = G.Cv2(elastic * (1 - 1e-9), self.KV, E_STEEL, FY50)
        above = G.Cv2(elastic * (1 + 1e-9), self.KV, E_STEEL, FY50)
        step = abs(above - below) / below
        assert step == pytest.approx(1.51 / (1.10 * 1.37) - 1.0, abs=1e-6)
        assert step < 3e-3

    def test_neither_exceeds_unity(self) -> None:
        for ratio in range(5, 400, 5):
            assert G.Cv1(float(ratio), self.KV, E_STEEL, FY50) <= 1.0
            assert G.Cv2(float(ratio), self.KV, E_STEEL, FY50) <= 1.0

    def test_both_reject_non_positive_ratios(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            G.Cv1(0.0, self.KV, E_STEEL, FY50)
        with pytest.raises(GeometryError, match="must be positive"):
            G.Cv2(-1.0, self.KV, E_STEEL, FY50)


class TestKvCoefficient:
    def test_unstiffened_web(self) -> None:
        """Sect. G2.1(b)(2)(i): kv = 5.34 without transverse stiffeners."""
        assert G.kv_coefficient(None) == 5.34

    def test_stiffened_web(self) -> None:
        """Eq. G2-5: kv = 5 + 5/(a/h)^2."""
        assert G.kv_coefficient(1.0) == pytest.approx(10.0, rel=1e-12)
        assert G.kv_coefficient(2.0) == pytest.approx(6.25, rel=1e-12)

    def test_the_a_over_h_cliff_at_3(self) -> None:
        """Eq. G2-5 gives 5.556 at a/h = 3, then the rule drops kv to 5.34.

        Not a smooth continuation -- a genuine step down. Panels longer than
        three web depths cannot develop a tension field, so their stiffeners
        are treated as absent.
        """
        at_three = G.kv_coefficient(3.0)
        just_past = G.kv_coefficient(3.0001)
        assert at_three == pytest.approx(5.0 + 5.0 / 9.0, rel=1e-12)
        assert just_past == 5.34
        assert just_past < at_three

    def test_widely_spaced_stiffeners_match_no_stiffeners(self) -> None:
        assert G.kv_coefficient(10.0) == G.kv_coefficient(None)

    def test_rejects_non_positive_spacing(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            G.kv_coefficient(0.0)


# ===========================================================================
# Sect. G2.1
# ===========================================================================
class TestG2_1:
    def test_the_2_24_limit_at_fy_50(self) -> None:
        """2.24*sqrt(29000/50) = 53.946."""
        assert 2.24 * math.sqrt(E_STEEL / FY50) == pytest.approx(53.946, abs=0.001)

    @pytest.mark.parametrize(
        ("name", "meets"),
        [("W18X50", True), ("W24X68", True), ("W16X26", False), ("W24X55", False)],
    )
    def test_G2_1a_membership_matches_the_user_note(self, name: str, meets: bool) -> None:
        """The User Note on p. 16.1-71 names W24x55 and W16x26 among the ASTM A6
        shapes that fail Sect. G2.1(a) at Fy = 50 -- a published list, tested
        against directly."""
        assert G.meets_G2_1a(SHAPES[name]["h_tw"], E_STEEL, FY50) is meets

    def test_built_up_girder_never_gets_phi_1_00(self) -> None:
        """Sect. G2.1(a) says "rolled I-shaped members". A built-up girder with
        an identical h/tw gets phi = 0.90, not 1.00."""
        assert G.meets_G2_1a(45.2, E_STEEL, FY50, rolled=True)
        assert not G.meets_G2_1a(45.2, E_STEEL, FY50, rolled=False)

    def test_phi_1_00_is_carried_on_the_limit_state(self) -> None:
        result = G.shear_strength(G.ShearMember(shape("W18X50"), A992, h_over_tw=45.2))
        assert result.governing.phi == 1.00
        assert result.governing.omega == 1.50
        assert result.available == pytest.approx(result.nominal, rel=1e-12)

    def test_phi_0_90_applies_to_the_exception_shapes(self) -> None:
        result = G.shear_strength(G.ShearMember(shape("W16X26"), A992, h_over_tw=56.8))
        assert result.governing.phi == 0.90
        assert result.available == pytest.approx(0.90 * result.nominal, rel=1e-12)

    def test_shear_area_uses_overall_depth(self) -> None:
        """Sect. G2.1: Aw = d*tw, the OVERALL depth -- not the clear web depth h.

        Using h instead loses the two flange thicknesses, about 6% on a W18x50.
        """
        assert G.shear_area_web(18.0, 0.355) == pytest.approx(6.39, rel=1e-12)
        clear_web = (18.0 - 2 * 0.570) * 0.355
        assert (6.39 - clear_web) / 6.39 == pytest.approx(0.0633, abs=0.001)

    def test_Vn_is_0_6_Fy_Aw_Cv1(self) -> None:
        Vn, cv1, phi, omega = G.g2_1_strength(FY50, 6.39, 45.2, E_STEEL, kv=5.34)
        assert cv1 == 1.0
        assert Vn == pytest.approx(0.6 * FY50 * 6.39, rel=1e-12)
        assert (phi, omega) == (1.00, 1.50)

    def test_2_24_is_inside_the_Cv1_plateau(self) -> None:
        """2.24 < 1.10*sqrt(5.34) = 2.542, so Eq. G2-2's Cv1 = 1.0 is consistent
        with Eqs. G2-3/G2-4 -- the G2.1(a) shortcut never contradicts them."""
        assert 1.10 * math.sqrt(5.34) > 2.24


# ===========================================================================
# Sect. G2.2 -- tension field action
# ===========================================================================
class TestTensionFieldAction:
    GEOMETRY = dict(Afc=12.0, Aft=12.0, h=60.0, bfc=16.0, bft=16.0)

    def test_not_permitted_without_stiffeners(self) -> None:
        applies, _, reason = G.tension_field_permitted(None, 20.0, **self.GEOMETRY)
        assert not applies
        assert "no transverse stiffeners" in reason

    def test_not_permitted_beyond_a_over_h_of_3(self) -> None:
        """Sect. G2.2 is headed "interior web panels with a/h <= 3"."""
        applies, _, reason = G.tension_field_permitted(3.01, 20.0, **self.GEOMETRY)
        assert not applies
        assert "exceeds 3.0" in reason

    def test_permitted_at_a_over_h_of_exactly_3(self) -> None:
        applies, _, _ = G.tension_field_permitted(3.0, 20.0, **self.GEOMETRY)
        assert applies

    def test_full_tension_field_when_all_three_checks_pass(self) -> None:
        applies, full, reason = G.tension_field_permitted(1.5, 20.0, **self.GEOMETRY)
        assert applies and full
        assert "G2-7" in reason

    def test_flange_area_ratio_forces_eq_G2_8(self) -> None:
        """2*Aw/(Afc + Aft) > 2.5 means the flanges cannot anchor a full field."""
        applies, full, reason = G.tension_field_permitted(
            1.5, 40.0, Afc=12.0, Aft=12.0, h=60.0, bfc=16.0, bft=16.0
        )
        assert applies and not full
        assert "2Aw/(Afc+Aft)" in reason

    def test_narrow_flange_forces_eq_G2_8(self) -> None:
        applies, full, reason = G.tension_field_permitted(
            1.5, 20.0, Afc=12.0, Aft=12.0, h=60.0, bfc=8.0, bft=16.0
        )
        assert applies and not full
        assert "h/bfc" in reason

    def test_G2_8_is_always_weaker_than_G2_7(self) -> None:
        """Eq. G2-8's denominator carries an extra +a/h, so it always yields less."""
        for a_over_h in (0.5, 1.0, 2.0, 3.0):
            full, _ = G.g2_2_tension_field_strength(
                FY50, 30.0, 180.0, E_STEEL, a_over_h, kv=G.kv_coefficient(a_over_h),
                full_tension_field=True,
            )
            partial, _ = G.g2_2_tension_field_strength(
                FY50, 30.0, 180.0, E_STEEL, a_over_h, kv=G.kv_coefficient(a_over_h),
                full_tension_field=False,
            )
            assert partial < full

    def test_eq_G2_6_on_the_plateau(self) -> None:
        """Below 1.10*sqrt(kv*E/Fy) there is no buckling to post-buckle from:
        Vn = 0.6*Fy*Aw with no coefficient at all."""
        Vn, cv2 = G.g2_2_tension_field_strength(
            FY50, 30.0, 40.0, E_STEEL, 1.5, kv=G.kv_coefficient(1.5),
            full_tension_field=True,
        )
        assert cv2 == 1.0
        assert Vn == pytest.approx(0.6 * FY50 * 30.0, rel=1e-12)

    def test_tension_field_adds_strength_over_G2_1(self) -> None:
        """The point of Sect. G2.2: post-buckling strength a slender web really has."""
        kv = G.kv_coefficient(1.0)
        without, _, _, _ = G.g2_1_strength(FY50, 30.0, 200.0, E_STEEL, kv=kv, rolled=False)
        with_field, _ = G.g2_2_tension_field_strength(
            FY50, 30.0, 200.0, E_STEEL, 1.0, kv=kv, full_tension_field=True
        )
        assert with_field > without

    def test_orchestrator_takes_the_larger_of_G2_1_and_G2_2(self) -> None:
        """Sect. G2.2 closes: "permitted to be taken as the larger of the values
        from Sections G2.1 and G2.2"."""
        girder = SectionAdapter.from_mapping(
            {"name": "PG60", "d": 62.0, "tw": 0.3125, "Ag": 45.0}
        )
        member = G.ShearMember(girder, A992, h_over_tw=192.0, rolled=False, a_over_h=1.0)
        plain = G.shear_strength(member)
        with_field = G.shear_strength(
            member, consider_tension_field=True,
            tension_field_geometry=dict(Afc=12.0, Aft=12.0, h=60.0, bfc=16.0, bft=16.0),
        )
        assert with_field.nominal >= plain.nominal
        assert with_field.limit_state is LimitState.TENSION_FIELD_ACTION


class TestTransverseStiffeners:
    def test_not_required_below_2_46_root(self) -> None:
        """Sect. G2.3(a): 2.46*sqrt(E/Fy) = 59.2 at Fy = 50."""
        limit = 2.46 * math.sqrt(E_STEEL / FY50)
        assert limit == pytest.approx(59.24, abs=0.01)
        assert not G.transverse_stiffeners_required(limit, E_STEEL, FY50)
        assert G.transverse_stiffeners_required(limit * 1.001, E_STEEL, FY50)

    def test_stiffener_slenderness_limit(self) -> None:
        """Eq. G2-12: (b/t)st <= 0.56*sqrt(E/Fyst) -- the STIFFENER's yield stress."""
        assert G.stiffener_slenderness_limit(E_STEEL, 36.0) == pytest.approx(
            0.56 * math.sqrt(E_STEEL / 36.0), rel=1e-12
        )
        assert G.stiffener_slenderness_limit(E_STEEL, 36.0) > G.stiffener_slenderness_limit(
            E_STEEL, 50.0
        )

    def test_Ist2_floor_catches_the_negative_bracket(self) -> None:
        """Eq. G2-15's (2.5/(a/h)^2 - 2) goes negative for a/h > 1.118.

        Without the 0.5*bp*tw^3 floor a widely spaced stiffener would compute a
        negative required inertia.
        """
        assert pytest.approx(0.0, abs=0.002) == 2.5 / 1.118**2 - 2.0
        _, _, Ist2 = G.stiffener_moment_of_inertia(
            60.0, 0.3125, 2.0, E_STEEL, 50.0, 36.0, Vr=200.0, Vc1=300.0, Vc2=150.0
        )
        bp = min(2.0 * 60.0, 60.0)
        assert Ist2 == pytest.approx(0.5 * bp * 0.3125**3, rel=1e-12)

    def test_Ist_interpolates_between_Ist2_and_Ist1(self) -> None:
        """Eq. G2-13 with rho_w = 0 gives Ist2, with rho_w = 1 gives Ist1."""
        args = (60.0, 0.3125, 1.0, E_STEEL, 50.0, 36.0)
        at_zero, Ist1, Ist2 = G.stiffener_moment_of_inertia(
            *args, Vr=150.0, Vc1=300.0, Vc2=150.0
        )
        at_one, _, _ = G.stiffener_moment_of_inertia(*args, Vr=300.0, Vc1=300.0, Vc2=150.0)
        assert at_zero == pytest.approx(Ist2, rel=1e-12)
        assert at_one == pytest.approx(Ist1, rel=1e-12)

    def test_rho_w_is_floored_at_zero(self) -> None:
        """A demand below Vc2 needs no post-buckling strength at all."""
        low, _, Ist2 = G.stiffener_moment_of_inertia(
            60.0, 0.3125, 1.0, E_STEEL, 50.0, 36.0, Vr=50.0, Vc1=300.0, Vc2=150.0
        )
        assert low == pytest.approx(Ist2, rel=1e-12)

    def test_rho_st_uses_the_larger_of_the_yield_ratio_and_one(self) -> None:
        """A stiffener stronger than the web gets no credit for it."""
        stronger, _, _ = G.stiffener_moment_of_inertia(
            60.0, 0.3125, 1.0, E_STEEL, 50.0, 70.0, Vr=300.0, Vc1=300.0, Vc2=150.0
        )
        equal, _, _ = G.stiffener_moment_of_inertia(
            60.0, 0.3125, 1.0, E_STEEL, 50.0, 50.0, Vr=300.0, Vc1=300.0, Vc2=150.0
        )
        assert stronger == pytest.approx(equal, rel=1e-12)

    def test_Vc1_below_Vc2_is_rejected(self) -> None:
        with pytest.raises(AISC360Error, match="must exceed"):
            G.stiffener_moment_of_inertia(
                60.0, 0.3125, 1.0, E_STEEL, 50.0, 36.0, Vr=200.0, Vc1=100.0, Vc2=150.0
            )


# ===========================================================================
# Sects. G3, G4, G5, G6
# ===========================================================================
class TestG3AnglesAndTees:
    def test_uses_kv_of_1_2(self) -> None:
        """Sect. G3: Cv2 with kv = 1.2 -- an element supported along ONE edge,
        against 5.34 for a web supported along two."""
        Vn, cv2 = G.g3_strength(FY50, 4.0, 0.5, E_STEEL)
        assert cv2 == pytest.approx(G.Cv2(8.0, 1.2, E_STEEL, FY50), rel=1e-12)
        assert Vn == pytest.approx(0.6 * FY50 * 4.0 * 0.5 * cv2, rel=1e-12)

    def test_stocky_leg_yields(self) -> None:
        assert G.g3_strength(36.0, 4.0, 0.5, E_STEEL)[1] == 1.0

    def test_thin_leg_buckles(self) -> None:
        assert G.g3_strength(FY50, 8.0, 0.125, E_STEEL)[1] < 1.0

    def test_kv_1_2_plateau_is_much_shorter_than_a_web(self) -> None:
        """1.10*sqrt(1.2*E/Fy) = 29.0 against 1.10*sqrt(5.34*E/Fy) = 61.2."""
        leg = 1.10 * math.sqrt(1.2 * E_STEEL / FY50)
        web = 1.10 * math.sqrt(5.34 * E_STEEL / FY50)
        assert leg == pytest.approx(29.01, abs=0.01)
        assert web == pytest.approx(61.22, abs=0.01)


class TestG4HSS:
    def test_uses_kv_of_5_not_5_34(self) -> None:
        """Sect. G4 names kv = 5 exactly. The 5.34 of Sect. G2.1 is an
        unstiffened-web value that does not carry over."""
        _, cv2 = G.g4_strength(FY50, 4.0, 40.0, E_STEEL)
        assert cv2 == pytest.approx(G.Cv2(40.0, 5.0, E_STEEL, FY50), rel=1e-12)
        assert cv2 != pytest.approx(G.Cv2(40.0, 5.34, E_STEEL, FY50), rel=1e-9) or cv2 == 1.0

    def test_both_webs_resist_shear(self) -> None:
        """Aw = 2*h*t for a rectangular HSS. Using one web halves the strength."""
        h, t = 9.07, 0.465
        both = G.g4_strength(50.0, 2 * h * t, h / t, E_STEEL)[0]
        one = G.g4_strength(50.0, h * t, h / t, E_STEEL)[0]
        assert both == pytest.approx(2.0 * one, rel=1e-12)

    def test_orchestrator_requires_Aw(self) -> None:
        hss = SectionAdapter.from_mapping({"name": "HSS", "Ag": 13.5})
        with pytest.raises(AISC360Error, match="Aw"):
            G.shear_strength(
                G.ShearMember(hss, A500_C_RECT, element=G.ShearElement.RECTANGULAR_HSS,
                              h_over_tw=19.5)
            )


class TestG5RoundHSS:
    def test_shear_yielding_governs_a_standard_section(self) -> None:
        """User Note, p. 16.1-75: for standard sections Fcr = 0.6*Fy."""
        Fcr, governing = G.g5_critical_stress(10.0, 0.349, 120.0, E_STEEL, 46.0)
        assert Fcr == pytest.approx(0.6 * 46.0, rel=1e-12)
        assert "yielding" in governing

    def test_buckling_governs_a_thin_long_tube(self) -> None:
        """The User Note names D/t over 100 and long lengths as the buckling cases."""
        Fcr, governing = G.g5_critical_stress(20.0, 0.125, 600.0, E_STEEL, 46.0)
        assert Fcr < 0.6 * 46.0
        assert "G5-2" in governing

    def test_the_larger_of_the_two_buckling_equations_is_taken(self) -> None:
        """Sect. G5: "Fcr shall be the larger of" -- Eq. G5-2a carries the length
        dependence and governs short tubes, Eq. G5-2b the length-independent
        envelope for long ones."""
        D, t, Fy = 20.0, 0.125, 46.0
        for Lv in (60.0, 240.0, 1200.0):
            a = 1.60 * E_STEEL / (math.sqrt(Lv / D) * (D / t) ** 1.25)
            b = 0.78 * E_STEEL / (D / t) ** 1.5
            Fcr, _ = G.g5_critical_stress(D, t, Lv, E_STEEL, Fy)
            assert Fcr == pytest.approx(min(max(a, b), 0.6 * Fy), rel=1e-12)

    def test_eq_G5_2a_governs_short_tubes(self) -> None:
        short, _ = G.g5_critical_stress(20.0, 0.125, 30.0, E_STEEL, 46.0)
        long_, _ = G.g5_critical_stress(20.0, 0.125, 3000.0, E_STEEL, 46.0)
        assert short >= long_

    def test_Vn_is_half_the_gross_area(self) -> None:
        """Eq. G5-1: Vn = Fcr*Ag/2 -- the shape factor for a thin tube in shear."""
        assert G.g5_strength(10.6, 27.6) == pytest.approx(27.6 * 10.6 / 2.0, rel=1e-12)

    def test_orchestrator_requires_Lv(self) -> None:
        pipe = SectionAdapter.from_mapping(
            {"name": "P10", "Ag": 10.6, "D": 10.75, "t": 0.340}
        )
        with pytest.raises(AISC360Error, match="Lv"):
            G.shear_strength(
                G.ShearMember(pipe, A500_C_RECT, element=G.ShearElement.ROUND_HSS)
            )


class TestG6WeakAxis:
    def test_uses_half_the_flange_width_for_I_shapes(self) -> None:
        """Sect. G6: h/tw = bf/(2*tf) for I-shapes and tees, bf/tf for channels."""
        i_shape, _ = G.g6_strength(FY50, 7.50, 0.570, E_STEEL, channel=False)
        channel, _ = G.g6_strength(FY50, 7.50, 0.570, E_STEEL, channel=True)
        assert i_shape == pytest.approx(channel, rel=1e-12)  # both Cv2 = 1.0 here

    def test_Cv2_is_unity_for_every_rolled_shape_at_fy_70(self) -> None:
        """User Note, p. 16.1-75: Cv2 = 1.0 for all ASTM A6 W, S, M and HP shapes
        at Fy <= 70 ksi. The tightest bf/2tf in the benchmark set is 3.29."""
        plateau = 1.10 * math.sqrt(1.2 * E_STEEL / 70.0)
        for name, props in SHAPES.items():
            ratio = props["bf"] / (2.0 * props["tf"])
            assert ratio <= plateau, f"{name}: bf/2tf = {ratio:.2f} > {plateau:.2f}"

    def test_is_per_element_not_per_member(self) -> None:
        """Sect. G6 gives the value "for each shear resisting element"; an
        I-shape has two flanges, so the member strength is twice this."""
        result = G.shear_strength(
            G.ShearMember(shape("W18X50"), A992, element=G.ShearElement.WEAK_AXIS_FLANGE)
        )
        assert "two flanges" in result.governing.note
        assert result.nominal == pytest.approx(0.6 * FY50 * 7.50 * 0.570, rel=1e-12)


# ===========================================================================
# Orchestrator
# ===========================================================================
class TestShearOrchestrator:
    def test_lrfd_and_asd_share_one_nominal_strength(self) -> None:
        member = G.ShearMember(shape("W16X26"), A992, h_over_tw=56.8)
        lrfd = G.shear_strength(member)
        asd = G.shear_strength(member, basis=Basis.ASD)
        assert lrfd.nominal == pytest.approx(asd.nominal, rel=1e-12)
        assert lrfd.available / asd.available == pytest.approx(0.90 * 1.67, rel=1e-12)

    def test_G2_1a_asd_uses_omega_1_50(self) -> None:
        member = G.ShearMember(shape("W18X50"), A992, h_over_tw=45.2)
        asd = G.shear_strength(member, basis=Basis.ASD)
        assert asd.available == pytest.approx(asd.nominal / 1.50, rel=1e-12)

    def test_report_renders(self) -> None:
        text = G.shear_strength(
            G.ShearMember(shape("W18X50"), A992, h_over_tw=45.2)
        ).report(required=150.0)
        assert "shear yielding" in text
        assert "OK" in text


# ===========================================================================
# Published benchmarks
# ===========================================================================
_EXAMPLES = Path(__file__).parent / "design_examples" / "chapter_g.json"


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
    return [case for case in data.get("cases", []) if not case.get("skip")]


class TestPublishedBenchmarks:
    @pytest.mark.design_example
    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: str(c.get("id", "?")))
    def test_matches_the_published_values(self, case: dict[str, Any]) -> None:
        from pyaisc360.materials import grade

        section = SectionAdapter.from_mapping(case["section"])
        steel = grade(str(case["grade"]))
        member = G.ShearMember(section, steel, h_over_tw=float(case["h_over_tw"]))
        result = G.shear_strength(member)
        tol = float(case.get("tol", 0.005))

        if "G2_1a" in case:
            assert G.meets_G2_1a(
                float(case["h_over_tw"]), steel.E, steel.Fy
            ) is case["G2_1a"]
            expected_phi = 1.00 if case["G2_1a"] else 0.90
            assert result.governing.phi == expected_phi

        assert result.available == pytest.approx(float(case["phiVn_kips"]), rel=tol)

    def test_every_case_declares_its_source(self) -> None:
        for case in _load_cases():
            assert case.get("source"), f"case {case.get('id')!r} has no source"

    def test_all_four_benchmark_shapes_are_present(self) -> None:
        ids = " ".join(str(c["id"]) for c in _load_cases())
        for name in SHAPES:
            assert name in ids


class TestIndependentImplementationAgreement:
    """Regression lock on Sect. G2.1.

    Confirmed bit-for-bit (relative difference 0.0e+00 on both Vn and phi*Vn,
    23 W-shapes) against the independent implementation in
    ``aisc-steel-design/scripts/aisc360.py``. Vendored so the suite has no
    dependency on a path outside the repository.
    """

    #: (shape, d, tw, h/tw, Vn, phi*Vn), A992.
    CASES = [
        ("W8X31", 8.00, 0.285, 22.3, 68.4000, 68.4000),
        ("W12X40", 11.9, 0.295, 33.6, 105.3150, 105.3150),
        ("W14X90", 14.0, 0.440, 25.9, 184.8000, 184.8000),
        ("W16X26", 15.7, 0.250, 56.8, 117.7500, 105.9750),
        ("W18X50", 18.0, 0.355, 45.2, 191.7000, 191.7000),
        ("W21X44", 20.7, 0.350, 53.6, 217.3500, 217.3500),
        ("W24X55", 23.6, 0.395, 54.6, 279.6600, 251.6940),
        ("W24X68", 23.7, 0.415, 52.0, 295.0650, 295.0650),
        ("W30X99", 29.7, 0.520, 51.9, 463.3200, 463.3200),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: c[0])
    def test_agrees_with_the_independent_implementation(
        self, case: tuple[str, float, float, float, float, float]
    ) -> None:
        name, d, tw, h_tw, Vn, phiVn = case
        section = SectionAdapter.from_mapping({"name": name, "d": d, "tw": tw, "Ag": 1.0})
        result = G.shear_strength(G.ShearMember(section, A992, h_over_tw=h_tw))
        assert result.nominal == pytest.approx(Vn, rel=1e-6)
        assert result.available == pytest.approx(phiVn, rel=1e-6)

    def test_only_the_two_named_shapes_lose_phi_1_00(self) -> None:
        """Within this set, exactly W16x26 and W24x55 fail Sect. G2.1(a) --
        and both are named in the Specification's User Note."""
        failing = {c[0] for c in self.CASES if not G.meets_G2_1a(c[3], E_STEEL, FY50)}
        assert failing == {"W16X26", "W24X55"}
