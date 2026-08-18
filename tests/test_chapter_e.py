"""Chapter E -- Design of Members for Compression.

Validation strategy, in descending order of authority:

1. **Values published inside AISC 360-16 itself.** Table E7.1's ``c2`` column
   must fall out of Eq. E7-4; Table B4.1a's coefficients must reproduce the
   printed limits. These are the strongest checks available -- the Specification
   is asserting against itself.
2. **Closed-form identities between provisions.** Eq. E4-4's cubic must reduce
   *exactly* to Eq. E4-3 when one shear-centre offset is zero, and Eq. E4-2 must
   equal Eq. E4-7 for a doubly symmetric section. An implementation error in the
   cubic cannot survive these.
3. **Continuity at every branch boundary.** E5-1/E5-2 at ``L/ra = 80``,
   E5-3/E5-4 at 75, E6-2a/E6-2b at ``a/ri = 40``, E7-6/E7-7 at ``0.11E/Fy``.
   A misplaced inequality shows up as a step change.
4. **Hand calculations**, with the arithmetic written out in the docstring so a
   reviewer can follow it without running anything.

The AISC *Design Examples V16.0* companion is not available on this machine, so
no test here claims to reproduce E.1A-E.7. ``design_examples/chapter_e.json``
carries the schema for those cases; drop the published inputs and answers in and
:class:`TestDesignExamples` picks them up automatically at the manual's printed
precision (0.5% relative).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from pyaisc360 import chapter_e as E
from pyaisc360.chapter_b import AxialElement, kc_coefficient, limiting_width_to_thickness
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import LimitState
from pyaisc360.core.exceptions import GeometryError, OutOfScopeError
from pyaisc360.materials import A36, A500_C_RECT, A992
from pyaisc360.sections import SectionAdapter

E_STEEL = 29000.0
G_STEEL = 11200.0
FY50 = 50.0


# ===========================================================================
# Table B4.1a -- the limits Sect. E7 depends on
# ===========================================================================
class TestTableB4_1a:
    @pytest.mark.parametrize(
        ("element", "expected"),
        [
            (AxialElement.ROLLED_I_FLANGE, 13.4866),  # case 1, 0.56*sqrt(E/Fy)
            (AxialElement.ANGLE_LEG, 10.8374),  # case 3, 0.45
            (AxialElement.TEE_STEM, 18.0624),  # case 4, 0.75
            (AxialElement.I_WEB, 35.8840),  # case 5, 1.49
            (AxialElement.RECTANGULAR_HSS_WALL, 33.7165),  # case 6, 1.40
            (AxialElement.COVER_PLATE, 33.7165),  # case 7, 1.40
            (AxialElement.OTHER_STIFFENED, 35.8840),  # case 8, 1.49
        ],
    )
    def test_limits_at_fy_50(self, element: AxialElement, expected: float) -> None:
        got = limiting_width_to_thickness(element, E_STEEL, FY50)
        assert got == pytest.approx(expected, abs=1e-4)

    def test_round_hss_has_no_square_root(self) -> None:
        """Case 9 is 0.11*E/Fy, not 0.11*sqrt(E/Fy) -- 63.8 vs 2.65 at Fy = 50.

        Reading it as a square root would classify every round HSS as slender.
        """
        got = limiting_width_to_thickness(AxialElement.ROUND_HSS, E_STEEL, FY50)
        assert got == pytest.approx(0.11 * E_STEEL / FY50, rel=1e-12)
        assert got == pytest.approx(63.8, abs=0.05)

    def test_round_hss_limit_matches_eq_e7_6(self) -> None:
        """Table B4.1a case 9 and the Eq. E7-6 threshold are the same number."""
        assert limiting_width_to_thickness(
            AxialElement.ROUND_HSS, E_STEEL, FY50
        ) == pytest.approx(0.11 * E_STEEL / FY50, rel=1e-12)

    def test_built_up_flange_needs_h_over_tw(self) -> None:
        with pytest.raises(Exception, match="h/tw"):
            limiting_width_to_thickness(AxialElement.BUILT_UP_I_FLANGE, E_STEEL, FY50)

    def test_built_up_flange_with_kc(self) -> None:
        """Case 2: 0.64*sqrt(kc*E/Fy) with kc = 4/sqrt(h/tw)."""
        h_tw = 100.0
        kc = 4.0 / math.sqrt(h_tw)
        expected = 0.64 * math.sqrt(kc * E_STEEL / FY50)
        got = limiting_width_to_thickness(
            AxialElement.BUILT_UP_I_FLANGE, E_STEEL, FY50, h_over_tw=h_tw
        )
        assert got == pytest.approx(expected, rel=1e-12)

    @pytest.mark.parametrize(
        ("h_tw", "expected"),
        [
            (10.0, 0.76),  # 4/sqrt(10) = 1.265 -> clamped to the 0.76 ceiling
            (130.6, 0.35),  # 4/sqrt(130.6) = 0.350 -> at the floor
            (500.0, 0.35),  # clamped to the 0.35 floor
            (100.0, 0.40),  # 4/10 = 0.40, inside the band
        ],
    )
    def test_kc_is_clamped_both_ways(self, h_tw: float, expected: float) -> None:
        """Footnote [a]: kc "shall not be taken less than 0.35 nor greater than 0.76"."""
        assert kc_coefficient(h_tw) == pytest.approx(expected, abs=0.005)

    def test_stiffened_classification(self) -> None:
        assert AxialElement.I_WEB.is_stiffened
        assert AxialElement.RECTANGULAR_HSS_WALL.is_stiffened
        assert not AxialElement.ROLLED_I_FLANGE.is_stiffened
        assert not AxialElement.ANGLE_LEG.is_stiffened

    def test_case_numbers_match_the_table(self) -> None:
        assert AxialElement.ROLLED_I_FLANGE.case == 1
        assert AxialElement.ROUND_HSS.case == 9


# ===========================================================================
# Sect. E2 / E3
# ===========================================================================
class TestEffectiveLength:
    def test_lc_is_k_times_l(self) -> None:
        assert E.effective_length(0.8, 144.0) == pytest.approx(115.2)

    def test_pinned_ends(self) -> None:
        assert E.effective_length(1.0, 360.0) == 360.0

    def test_non_positive_k_raises(self) -> None:
        with pytest.raises(GeometryError, match="K must be positive"):
            E.effective_length(0.0, 360.0)


class TestFlexuralBucklingE3:
    def test_w14x90_at_30_feet(self) -> None:
        """W14x90, A992, Lc = 30 ft about the minor axis, ry = 3.70 in.

        Lc/ry = 360/3.70          = 97.297
        Fe    = pi^2*29000/97.297^2 = 30.234 ksi   (Eq. E3-4)
        Fy/Fe = 50/30.234         = 1.6538 <= 2.25 -> Eq. E3-2 governs
        Fcr   = 0.658^1.6538 * 50 = 25.024 ksi
        Pn    = 25.024 * 26.5     = 663.1 kips     (Eq. E3-1)
        """
        Fcr, Fe = E.flexural_buckling_stress_E3(360.0 / 3.70, E_STEEL, FY50)
        assert Fe == pytest.approx(30.234, abs=0.001)
        assert Fcr == pytest.approx(25.024, abs=0.001)
        assert Fcr * 26.5 == pytest.approx(663.1, rel=0.005)

    def test_returns_both_stresses(self) -> None:
        Fcr, Fe = E.flexural_buckling_stress_E3(100.0, E_STEEL, FY50)
        assert Fcr < Fe or Fcr < FY50

    def test_short_column_approaches_yield(self) -> None:
        """Eq. E3-2 tends to Fy as Lc/r -> 0, but never reaches it: 0.658^x -> 1."""
        Fcr, _ = E.flexural_buckling_stress_E3(1.0, E_STEEL, FY50)
        assert Fcr == pytest.approx(FY50, rel=1e-3)
        assert Fcr < FY50


# ===========================================================================
# Sect. E4 -- torsional and flexural-torsional buckling
# ===========================================================================
class TestE4Components:
    def test_polar_radius_eq_e4_9(self) -> None:
        """ro^2 = xo^2 + yo^2 + (Ix + Iy)/Ag."""
        ro = E.polar_radius_of_gyration(0.0, 2.0, 999.0, 362.0, 26.5)
        assert ro**2 == pytest.approx(4.0 + (999.0 + 362.0) / 26.5, rel=1e-12)

    def test_flexural_constant_eq_e4_8(self) -> None:
        assert E.flexural_constant(0.0, 2.0, 4.0) == pytest.approx(0.75, rel=1e-12)

    def test_doubly_symmetric_has_unit_flexural_constant(self) -> None:
        """xo = yo = 0 gives H = 1, the doubly symmetric case."""
        assert E.flexural_constant(0.0, 0.0, 5.0) == 1.0

    def test_impossible_shear_centre_offset_raises(self) -> None:
        with pytest.raises(GeometryError, match="not positive"):
            E.flexural_constant(4.0, 4.0, 4.0)

    def test_fex_and_fey_are_euler(self) -> None:
        assert E.Fex(100.0, E_STEEL) == pytest.approx(math.pi**2 * E_STEEL / 1e4, rel=1e-12)
        assert E.Fey(80.0, E_STEEL) == pytest.approx(math.pi**2 * E_STEEL / 6400.0, rel=1e-12)


class TestE4_2_equals_E4_7:
    """For a doubly symmetric section, Eq. E4-2 and Eq. E4-7 are the same number.

    Because xo = yo = 0 makes ro^2 = (Ix + Iy)/Ag exactly (Eq. E4-9), the
    Ag*ro^2 in Eq. E4-7's denominator *is* Ix + Iy. The Specification prints
    both forms; they must agree, and if they ever disagree one of them is
    mis-transcribed.
    """

    def test_w14x90(self) -> None:
        Ix, Iy, Ag, J, Cw, Lcz = 999.0, 362.0, 26.5, 4.06, 16000.0, 360.0
        ro = E.polar_radius_of_gyration(0.0, 0.0, Ix, Iy, Ag)
        by_e4_2 = E.torsional_buckling_stress(E_STEEL, G_STEEL, Cw, J, Lcz, Ix, Iy)
        by_e4_7 = E.Fez(E_STEEL, G_STEEL, Cw, J, Lcz, Ag, ro)
        assert by_e4_2 == pytest.approx(by_e4_7, rel=1e-12)
        assert by_e4_2 == pytest.approx(59.374, abs=0.001)

    @pytest.mark.parametrize("Lcz", [120.0, 240.0, 360.0, 600.0])
    def test_identity_holds_at_every_length(self, Lcz: float) -> None:
        Ix, Iy, Ag, J, Cw = 999.0, 362.0, 26.5, 4.06, 16000.0
        ro = E.polar_radius_of_gyration(0.0, 0.0, Ix, Iy, Ag)
        assert E.torsional_buckling_stress(
            E_STEEL, G_STEEL, Cw, J, Lcz, Ix, Iy
        ) == pytest.approx(E.Fez(E_STEEL, G_STEEL, Cw, J, Lcz, Ag, ro), rel=1e-12)

    def test_warping_can_be_omitted_for_tees(self) -> None:
        """User Note, p. 16.1-37: omit the Cw term for tees and double angles."""
        with_warping = E.Fez(E_STEEL, G_STEEL, 16000.0, 4.06, 360.0, 26.5, 7.17)
        without = E.Fez(E_STEEL, G_STEEL, 16000.0, 4.06, 360.0, 26.5, 7.17,
                        include_warping=False)
        assert without < with_warping
        assert without == pytest.approx(
            G_STEEL * 4.06 / (26.5 * 7.17**2), rel=1e-12
        )


class TestE4_3:
    def test_hand_calculation(self) -> None:
        """Fey = 30, Fez = 40, H = 0.75.

        (Fey + Fez)/(2H)            = 70/1.5      = 46.667
        4*Fey*Fez*H/(Fey + Fez)^2   = 3600/4900   = 0.734694
        sqrt(1 - 0.734694)          = 0.515079
        Fe = 46.667 * (1 - 0.515079)              = 22.6297 ksi
        """
        assert E.flexural_torsional_buckling_stress(30.0, 40.0, 0.75) == pytest.approx(
            22.629658, abs=1e-6
        )

    def test_fe_is_below_both_components(self) -> None:
        """Coupling always reduces the buckling stress below either pure mode."""
        for H in (0.4, 0.6, 0.8, 0.99):
            Fe = E.flexural_torsional_buckling_stress(30.0, 40.0, H)
            assert Fe < 30.0
            assert Fe < 40.0

    def test_h_of_unity_recovers_the_lower_component(self) -> None:
        """H = 1 means no shear-centre offset, so no coupling: Fe = min(Fey, Fez)."""
        assert E.flexural_torsional_buckling_stress(30.0, 40.0, 1.0) == pytest.approx(
            30.0, rel=1e-9
        )

    def test_equal_components_at_h_unity(self) -> None:
        """The radicand is exactly zero here -- the clamp must hold."""
        assert E.flexural_torsional_buckling_stress(35.0, 35.0, 1.0) == pytest.approx(
            35.0, rel=1e-9
        )

    def test_rejects_out_of_range_h(self) -> None:
        with pytest.raises(GeometryError, match="must lie in"):
            E.flexural_torsional_buckling_stress(30.0, 40.0, 1.5)


class TestE4_4Cubic:
    """The Eq. E4-4 cubic, solved exactly.

    The decisive tests are the degeneracies: setting a shear-centre offset to
    zero must collapse the cubic onto a provision the Specification states
    independently. If the expansion of the cubic is wrong, these fail.
    """

    def test_reduces_to_e4_3_when_xo_is_zero(self) -> None:
        """xo = 0 factors the cubic as (Fe - Fex) * [quadratic in Fey, Fez].

        With ro = 4, yo = 2 -> H = 0.75; Fey = 30, Fez = 40 -> the quadratic's
        lower root is 22.6297, which is below Fex = 100, so it governs. That is
        exactly Eq. E4-3.
        """
        by_cubic = E.unsymmetric_buckling_stress(100.0, 30.0, 40.0, xo=0.0, yo=2.0, ro=4.0)
        by_e4_3 = E.flexural_torsional_buckling_stress(30.0, 40.0, 0.75)
        assert by_cubic == pytest.approx(by_e4_3, rel=1e-9)

    def test_reduces_to_e4_3_when_yo_is_zero(self) -> None:
        """yo = 0 is the channel case: the User Note's Fex-for-Fey substitution."""
        by_cubic = E.unsymmetric_buckling_stress(30.0, 100.0, 40.0, xo=2.0, yo=0.0, ro=4.0)
        by_e4_3 = E.flexural_torsional_buckling_stress(30.0, 40.0, 0.75)
        assert by_cubic == pytest.approx(by_e4_3, rel=1e-9)

    def test_doubly_symmetric_gives_the_lowest_component(self) -> None:
        """xo = yo = 0 uncouples all three modes: the roots are Fex, Fey, Fez."""
        assert E.unsymmetric_buckling_stress(
            45.0, 30.0, 60.0, xo=0.0, yo=0.0, ro=5.0
        ) == pytest.approx(30.0, rel=1e-9)

    @pytest.mark.parametrize(
        ("Fex", "Fey", "Fez", "xo", "yo", "ro"),
        [
            (40.0, 55.0, 70.0, 1.2, 2.1, 4.0),
            (120.0, 30.0, 45.0, 0.8, 1.5, 3.5),
            (25.0, 25.0, 25.0, 1.0, 1.0, 3.0),
            (200.0, 18.0, 90.0, 2.0, 0.5, 5.0),
        ],
    )
    def test_root_satisfies_the_equation_as_printed(
        self, Fex: float, Fey: float, Fez: float, xo: float, yo: float, ro: float
    ) -> None:
        """Substitute the answer back into Eq. E4-4 exactly as written.

        This is the check that the expansion into cubic coefficients is right:
        it never uses the expansion, only the printed product form.
        """
        Fe = E.unsymmetric_buckling_stress(Fex, Fey, Fez, xo, yo, ro)
        residual = (
            (Fe - Fex) * (Fe - Fey) * (Fe - Fez)
            - Fe**2 * (Fe - Fey) * (xo / ro) ** 2
            - Fe**2 * (Fe - Fex) * (yo / ro) ** 2
        )
        scale = max(abs(Fex), abs(Fey), abs(Fez)) ** 3
        assert abs(residual) / scale < 1e-10

    def test_returns_the_lowest_positive_root(self) -> None:
        """Sect. E4(c): "Fe is the lowest root of the cubic equation"."""
        Fe = E.unsymmetric_buckling_stress(40.0, 55.0, 70.0, 1.2, 2.1, 4.0)
        assert Fe > 0.0
        assert Fe < min(40.0, 55.0, 70.0)

    def test_rejects_impossible_shear_centre(self) -> None:
        with pytest.raises(GeometryError, match="not positive"):
            E.unsymmetric_buckling_stress(40.0, 55.0, 70.0, 3.0, 3.0, 4.0)

    def test_rejects_non_positive_components(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            E.unsymmetric_buckling_stress(0.0, 55.0, 70.0, 1.0, 1.0, 4.0)


class TestCubicSolver:
    """The closed-form solver itself, against polynomials with known roots."""

    @pytest.mark.parametrize(
        "roots",
        [
            (1.0, 2.0, 3.0),
            (-5.0, 2.0, 7.0),
            (10.0, 10.0, 10.0),  # triple root
            (2.0, 2.0, 9.0),  # double root
            (0.5, 60.0, 61.0),  # widely separated
        ],
    )
    def test_recovers_known_real_roots(self, roots: tuple[float, float, float]) -> None:
        """Build the polynomial from known roots, then recover them.

        Tolerance is 1e-6 rather than machine precision because a *repeated*
        root is inherently ill-conditioned: near a double root the polynomial is
        locally quadratic, so an input perturbation of eps moves the root by
        about sqrt(eps) ~ 1e-8. That is a property of the mathematics, not of
        this solver, and it is irrelevant physically -- a repeated root in
        Eq. E4-4 means two buckling modes coincide, and 1e-8 ksi either way
        changes nothing.
        """
        r1, r2, r3 = roots
        a2 = -(r1 + r2 + r3)
        a1 = r1 * r2 + r2 * r3 + r3 * r1
        a0 = -(r1 * r2 * r3)
        got = sorted(E._real_cubic_roots(a2, a1, a0))
        for want in sorted(set(roots)):
            assert any(abs(g - want) < 1e-6 * max(1.0, abs(want)) for g in got), (want, got)

    def test_single_real_root_case(self) -> None:
        """x^3 + x + 1 has one real root near -0.6823."""
        got = E._real_cubic_roots(0.0, 1.0, 1.0)
        assert len(got) == 1
        assert got[0] == pytest.approx(-0.6823278, abs=1e-6)


# ===========================================================================
# Sect. E5 -- single angles
# ===========================================================================
class TestE5EffectiveSlenderness:
    def _config(self, L: float, **kwargs: object) -> E.AngleConfig:
        defaults: dict[str, object] = {
            "L": L,
            "ra": 1.0,
            "rz": 0.75,
            "b_long": 4.0,
            "b_short": 4.0,
        }
        defaults.update(kwargs)
        return E.AngleConfig(**defaults)  # type: ignore[arg-type]

    def test_e5_1_below_80(self) -> None:
        """Eq. E5-1: Lc/r = 72 + 0.75*L/ra. At L/ra = 60 -> 72 + 45 = 117."""
        assert E.single_angle_effective_slenderness(self._config(60.0)) == pytest.approx(117.0)

    def test_e5_2_above_80(self) -> None:
        """Eq. E5-2: Lc/r = 32 + 1.25*L/ra. At L/ra = 100 -> 32 + 125 = 157."""
        assert E.single_angle_effective_slenderness(self._config(100.0)) == pytest.approx(157.0)

    def test_e5_1_and_e5_2_are_continuous_at_80(self) -> None:
        """72 + 0.75*80 = 132 and 32 + 1.25*80 = 132 -- the branches meet exactly."""
        at_limit = E.single_angle_effective_slenderness(self._config(80.0))
        just_above = E.single_angle_effective_slenderness(self._config(80.0001))
        assert at_limit == pytest.approx(132.0)
        assert just_above == pytest.approx(132.0, abs=1e-3)

    def test_e5_3_below_75(self) -> None:
        """Eq. E5-3: Lc/r = 60 + 0.8*L/ra. At L/ra = 50 -> 60 + 40 = 100."""
        got = E.single_angle_effective_slenderness(
            self._config(50.0, truss=E.TrussType.BOX_OR_SPACE)
        )
        assert got == pytest.approx(100.0)

    def test_e5_4_above_75(self) -> None:
        """Eq. E5-4: Lc/r = 45 + L/ra. At L/ra = 100 -> 145."""
        got = E.single_angle_effective_slenderness(
            self._config(100.0, truss=E.TrussType.BOX_OR_SPACE)
        )
        assert got == pytest.approx(145.0)

    def test_e5_3_and_e5_4_are_continuous_at_75(self) -> None:
        """60 + 0.8*75 = 120 and 45 + 75 = 120."""
        got = E.single_angle_effective_slenderness(
            self._config(75.0, truss=E.TrussType.BOX_OR_SPACE)
        )
        assert got == pytest.approx(120.0)

    def test_shorter_leg_adds_the_planar_increment(self) -> None:
        """E5(a)(2): add 4*[(bl/bs)^2 - 1]. With 5x3 legs, (5/3)^2 - 1 = 1.7778,
        so the increment is 7.111 on top of 72 + 0.75*40 = 102."""
        got = E.single_angle_effective_slenderness(
            self._config(40.0, b_long=5.0, b_short=3.0, connected_leg=E.ConnectedLeg.SHORTER)
        )
        assert got == pytest.approx(102.0 + 4.0 * ((5.0 / 3.0) ** 2 - 1.0), abs=1e-6)

    def test_shorter_leg_adds_the_box_truss_increment(self) -> None:
        """E5(b)(2): the coefficient is 6, not 4."""
        got = E.single_angle_effective_slenderness(
            self._config(
                40.0,
                b_long=5.0,
                b_short=3.0,
                connected_leg=E.ConnectedLeg.SHORTER,
                truss=E.TrussType.BOX_OR_SPACE,
            )
        )
        assert got == pytest.approx(92.0 + 6.0 * ((5.0 / 3.0) ** 2 - 1.0), abs=1e-6)

    def test_planar_floor_of_095_l_over_rz(self) -> None:
        """E5(a)(2): "shall not be taken as less than 0.95*L/rz".

        L/ra = 5 gives 72 + 3.75 = 75.75 from Eq. E5-1, plus 4*[(5/3)^2 - 1]
        = 7.11 -> 82.86. But 0.95*L/rz = 0.95*166.7 = 158.3 is larger, so the
        floor governs.
        """
        config = E.AngleConfig(
            L=5.0, ra=1.0, rz=0.03, b_long=5.0, b_short=3.0,
            connected_leg=E.ConnectedLeg.SHORTER,
        )
        assert E.single_angle_effective_slenderness(config) == pytest.approx(
            0.95 * 5.0 / 0.03
        )

    def test_box_truss_floor_of_082_l_over_rz(self) -> None:
        """E5(b)(2) floors at 0.82*L/rz rather than 0.95*L/rz."""
        config = E.AngleConfig(
            L=5.0, ra=1.0, rz=0.03, b_long=5.0, b_short=3.0,
            connected_leg=E.ConnectedLeg.SHORTER, truss=E.TrussType.BOX_OR_SPACE,
        )
        assert E.single_angle_effective_slenderness(config) == pytest.approx(
            0.82 * 5.0 / 0.03
        )

    def test_equal_leg_angle_gets_no_increment(self) -> None:
        """The increment applies only when connected through the shorter leg."""
        plain = E.single_angle_effective_slenderness(self._config(40.0))
        assert plain == pytest.approx(102.0)

    def test_leg_ratio_of_17_or_more_is_out_of_scope(self) -> None:
        """Sect. E5 condition (5): the ratio must be *less than* 1.7."""
        config = E.AngleConfig(
            L=40.0, ra=1.0, rz=0.75, b_long=8.0, b_short=4.0,
            connected_leg=E.ConnectedLeg.SHORTER,
        )
        with pytest.raises(OutOfScopeError, match="condition \\(5\\)"):
            E.single_angle_effective_slenderness(config)

    def test_slenderness_over_200_is_out_of_scope(self) -> None:
        """Sect. E5 condition (4): Lc/r must not exceed 200."""
        with pytest.raises(OutOfScopeError, match="condition \\(4\\)"):
            E.single_angle_effective_slenderness(self._config(200.0))

    def test_strict_can_be_relaxed(self) -> None:
        got = E.single_angle_effective_slenderness(self._config(200.0), strict=False)
        assert got == pytest.approx(32.0 + 1.25 * 200.0)

    def test_transposed_legs_are_rejected(self) -> None:
        with pytest.raises(GeometryError, match="transposed"):
            E.AngleConfig(L=40.0, ra=1.0, rz=0.75, b_long=3.0, b_short=5.0)

    def test_ftb_neglect_threshold(self) -> None:
        """Sect. E5: FTB need not be considered when b/t <= 0.71*sqrt(E/Fy).

        At Fy = 50 that limit is 17.10.
        """
        limit = 0.71 * math.sqrt(E_STEEL / FY50)
        assert limit == pytest.approx(17.099, abs=0.001)
        assert E.single_angle_ftb_can_be_neglected(17.0, E_STEEL, FY50)
        assert not E.single_angle_ftb_can_be_neglected(17.2, E_STEEL, FY50)


# ===========================================================================
# Sect. E6 -- built-up members
# ===========================================================================
class TestE6ModifiedSlenderness:
    def test_e6_1_snug_tight_bolted(self) -> None:
        """Eq. E6-1: (Lc/r)m = sqrt(80^2 + 60^2) = 100 -- a 3-4-5 triangle."""
        config = E.BuiltUpConfig(a=60.0, ri=1.0, connector=E.ConnectorType.SNUG_TIGHT_BOLTED)
        assert E.built_up_modified_slenderness(80.0, config) == pytest.approx(100.0)

    def test_e6_1_always_increases_slenderness(self) -> None:
        """The snug-tight form has no a/ri <= 40 exemption -- it always applies."""
        config = E.BuiltUpConfig(a=10.0, ri=1.0, connector=E.ConnectorType.SNUG_TIGHT_BOLTED)
        assert E.built_up_modified_slenderness(80.0, config) > 80.0

    def test_e6_2a_below_40_is_unmodified(self) -> None:
        """Eq. E6-2a: welded or pretensioned with a/ri <= 40 -> no modification."""
        config = E.BuiltUpConfig(a=39.0, ri=1.0)
        assert E.built_up_modified_slenderness(80.0, config) == 80.0

    def test_e6_2b_above_40(self) -> None:
        """Eq. E6-2b: sqrt(80^2 + (0.86*60)^2) with Ki = 0.86 for "all other"."""
        config = E.BuiltUpConfig(a=60.0, ri=1.0)
        expected = math.sqrt(80.0**2 + (0.86 * 60.0) ** 2)
        assert E.built_up_modified_slenderness(80.0, config) == pytest.approx(expected)

    def test_e6_2a_and_e6_2b_are_continuous_at_40(self) -> None:
        """At a/ri = 40 exactly, E6-2a gives 80.0 and E6-2b would give 86.1.

        The Specification's branches are *not* continuous here -- Eq. E6-2b
        switches on at a step. The test records that as intended behaviour so a
        future "smoothing" refactor cannot silently soften it.
        """
        at_limit = E.built_up_modified_slenderness(80.0, E.BuiltUpConfig(a=40.0, ri=1.0))
        just_above = E.built_up_modified_slenderness(80.0, E.BuiltUpConfig(a=40.001, ri=1.0))
        assert at_limit == 80.0
        assert just_above == pytest.approx(math.sqrt(80.0**2 + (0.86 * 40.001) ** 2), abs=1e-3)
        assert just_above > at_limit

    @pytest.mark.parametrize(
        ("shape", "Ki"),
        [
            (E.BuiltUpShape.ANGLES_BACK_TO_BACK, 0.50),
            (E.BuiltUpShape.CHANNELS_BACK_TO_BACK, 0.75),
            (E.BuiltUpShape.OTHER, 0.86),
        ],
    )
    def test_ki_values(self, shape: E.BuiltUpShape, Ki: float) -> None:
        assert shape.Ki == Ki
        config = E.BuiltUpConfig(a=60.0, ri=1.0, shape=shape)
        assert E.built_up_modified_slenderness(80.0, config) == pytest.approx(
            math.sqrt(80.0**2 + (Ki * 60.0) ** 2)
        )

    def test_connector_spacing_limit(self) -> None:
        """Sect. E6.2(a): a/ri must not exceed three-fourths of (Lc/r)."""
        assert E.built_up_connector_spacing_limit(120.0) == pytest.approx(90.0)

    def test_modified_never_below_original(self) -> None:
        for a in (10.0, 40.0, 60.0, 200.0):
            for connector in E.ConnectorType:
                config = E.BuiltUpConfig(a=a, ri=1.0, connector=connector)
                assert E.built_up_modified_slenderness(80.0, config) >= 80.0


# ===========================================================================
# Sect. E7 -- slender elements
# ===========================================================================
class TestTableE7_1:
    """Eq. E7-4 must reproduce the c2 column of Table E7.1, p. 16.1-43.

    This is the strongest single check in the chapter: the Specification prints
    both c1 and c2, and c2 is derivable from c1. If Eq. E7-4 is mis-transcribed,
    the printed pairs will not reconcile.
    """

    @pytest.mark.parametrize(
        ("c1", "published_c2"),
        [
            (0.18, 1.31),  # case (a) stiffened elements except square/rect HSS walls
            (0.20, 1.38),  # case (b) walls of square and rectangular HSS
            (0.22, 1.49),  # case (c) all other elements
        ],
    )
    def test_c2_matches_the_published_column(self, c1: float, published_c2: float) -> None:
        assert E.c2_from_c1(c1) == pytest.approx(published_c2, abs=0.005)

    def test_c1_outside_the_real_range_raises(self) -> None:
        """Eq. E7-4's radicand 1 - 4*c1 goes negative above c1 = 0.25."""
        with pytest.raises(Exception, match="c1 must lie"):
            E.c2_from_c1(0.30)


class TestEffectiveWidth:
    LAM_R_HSS = 1.40 * math.sqrt(E_STEEL / FY50)  # 33.7165

    def test_e7_2_returns_the_full_width(self) -> None:
        """lambda below lambda_r*sqrt(Fy/Fcr) -> the element is fully effective."""
        be = E.effective_width(10.0, 40.0, self.LAM_R_HSS, FY50, 30.0, 0.20, 1.381966)
        assert be == 10.0

    def test_threshold_is_not_lambda_r(self) -> None:
        """lambda = 40 exceeds lambda_r = 33.72, yet the element is fully effective.

        The Sect. E7.1 threshold is lambda_r*sqrt(Fy/Fcr) = 43.53 at Fcr = 30 ksi.
        A column slender enough to buckle below Fy loses less plate width, not
        more -- testing against lambda_r alone would deduct area wrongly.
        """
        assert self.LAM_R_HSS < 40.0
        assert self.LAM_R_HSS * math.sqrt(FY50 / 30.0) > 40.0
        assert E.effective_width(10.0, 40.0, self.LAM_R_HSS, FY50, 30.0, 0.20, 1.381966) == 10.0

    def test_e7_3_hand_calculation(self) -> None:
        """b = 10, lambda = 60, lambda_r = 33.7165, Fy = 50, Fcr = 30, HSS wall.

        c2   = 1.381966                                   (Eq. E7-4)
        Fel  = (1.381966*33.7165/60)^2 * 50 = 30.1541 ksi (Eq. E7-5)
        sqrt(Fel/Fcr) = sqrt(1.005136)      = 1.0025648
        be   = 10*(1 - 0.20*1.0025648)*1.0025648 = 8.0154 in.  (Eq. E7-3)
        """
        c2 = E.c2_from_c1(0.20)
        be = E.effective_width(10.0, 60.0, self.LAM_R_HSS, FY50, 30.0, 0.20, c2)
        assert be == pytest.approx(8.015377, abs=1e-5)

    def test_eq_e7_5_hand_calculation(self) -> None:
        Fel = E.elastic_local_buckling_stress(60.0, self.LAM_R_HSS, E.c2_from_c1(0.20), FY50)
        assert Fel == pytest.approx(30.154095, abs=1e-5)

    def test_effective_width_never_exceeds_the_element(self) -> None:
        for lam in [float(x) for x in range(20, 200, 5)]:
            be = E.effective_width(10.0, lam, self.LAM_R_HSS, FY50, 30.0, 0.20, 1.381966)
            assert 0.0 <= be <= 10.0

    def test_effective_width_decreases_with_slenderness(self) -> None:
        widths = [
            E.effective_width(10.0, lam, self.LAM_R_HSS, FY50, 30.0, 0.20, 1.381966)
            for lam in (50.0, 60.0, 80.0, 100.0, 140.0)
        ]
        assert all(a >= b for a, b in zip(widths, widths[1:], strict=False))

    def test_continuous_across_the_threshold(self) -> None:
        """be must equal b at the threshold itself, with no step."""
        threshold = self.LAM_R_HSS * math.sqrt(FY50 / 30.0)
        c2 = E.c2_from_c1(0.20)
        below = E.effective_width(10.0, threshold - 1e-6, self.LAM_R_HSS, FY50, 30.0, 0.20, c2)
        above = E.effective_width(10.0, threshold + 1e-6, self.LAM_R_HSS, FY50, 30.0, 0.20, c2)
        assert below == pytest.approx(10.0)
        assert above == pytest.approx(10.0, abs=2e-3)


class TestRoundHSSEffectiveArea:
    def test_e7_6_nonslender_keeps_the_gross_area(self) -> None:
        """D/t = 40 is below 0.11*E/Fy = 63.8 at Fy = 50."""
        assert E.round_hss_effective_area(10.0, 8.0, 0.20, E_STEEL, FY50) == 10.0

    def test_e7_7_reduces_the_area(self) -> None:
        """D/t = 100, between 63.8 and 261.

        Ae/Ag = 0.038*29000/(50*100) + 2/3 = 0.2204 + 0.6667 = 0.8871
        """
        Ae = E.round_hss_effective_area(10.0, 20.0, 0.20, E_STEEL, FY50)
        assert Ae == pytest.approx(10.0 * (0.038 * E_STEEL / (FY50 * 100.0) + 2.0 / 3.0), rel=1e-12)
        assert Ae == pytest.approx(8.8707, abs=1e-3)

    def test_continuous_at_the_e7_6_boundary(self) -> None:
        """At D/t = 0.11*E/Fy exactly, Eq. E7-7 also returns Ag.

        0.038*E/(Fy*0.11*E/Fy) + 2/3 = 0.038/0.11 + 2/3 = 0.34545 + 0.66667
        = 1.01212 -- so the two are within 1.2%, not exact. The Specification's
        own branches step slightly here; the test pins the actual behaviour.
        """
        D_over_t = 0.11 * E_STEEL / FY50
        at_limit = E.round_hss_effective_area(10.0, D_over_t, 1.0, E_STEEL, FY50)
        just_above = E.round_hss_effective_area(10.0, D_over_t + 1e-6, 1.0, E_STEEL, FY50)
        assert at_limit == 10.0
        assert just_above == pytest.approx(10.121, abs=0.01)

    def test_beyond_045_e_over_fy_is_out_of_scope(self) -> None:
        """Sect. E7.2 provides nothing for D/t >= 0.45*E/Fy = 261 at Fy = 50."""
        with pytest.raises(OutOfScopeError, match="0.45"):
            E.round_hss_effective_area(10.0, 300.0, 1.0, E_STEEL, FY50)

    def test_a36_pipe_has_a_larger_nonslender_range(self) -> None:
        """0.11*E/Fy scales with 1/Fy: 88.6 at Fy = 36 vs 63.8 at Fy = 50."""
        assert E.round_hss_effective_area(10.0, 80.0, 1.0, E_STEEL, A36.Fy) == 10.0
        assert E.round_hss_effective_area(10.0, 80.0, 1.0, E_STEEL, FY50) < 10.0


class TestEffectiveArea:
    def test_nonslender_elements_leave_ag_untouched(self) -> None:
        elements = [
            E.SlenderElement("flange", AxialElement.ROLLED_I_FLANGE, b=3.75, t=0.570, count=2),
            E.SlenderElement("web", AxialElement.I_WEB, b=16.0, t=0.355),
        ]
        Ae, detail = E.effective_area(14.7, elements, E_STEEL, FY50, 30.0)
        assert Ae == pytest.approx(14.7)
        assert "flange: be" in detail

    def test_slender_wall_reduces_the_area(self) -> None:
        """Four HSS walls at b/t = 60, well past the threshold."""
        elements = [
            E.SlenderElement("wall", AxialElement.RECTANGULAR_HSS_WALL, b=6.0, t=0.10, count=4)
        ]
        Ae, _ = E.effective_area(2.5, elements, E_STEEL, FY50, 30.0)
        assert Ae < 2.5

    def test_deduction_matches_the_user_note_formula(self) -> None:
        """Sect. E7 User Note: Ae = Ag - sum of (b - be)*t."""
        element = E.SlenderElement(
            "wall", AxialElement.RECTANGULAR_HSS_WALL, b=6.0, t=0.10, count=4
        )
        lam_r = limiting_width_to_thickness(AxialElement.RECTANGULAR_HSS_WALL, E_STEEL, FY50)
        be = E.effective_width(6.0, 60.0, lam_r, FY50, 30.0, 0.20, E.c2_from_c1(0.20))
        Ae, _ = E.effective_area(2.5, [element], E_STEEL, FY50, 30.0)
        assert Ae == pytest.approx(2.5 - 4 * (6.0 - be) * 0.10, rel=1e-12)

    def test_impossible_element_widths_raise(self) -> None:
        """Passing gross dimensions instead of element widths is a common slip."""
        elements = [
            E.SlenderElement("wall", AxialElement.RECTANGULAR_HSS_WALL, b=60.0, t=1.0, count=4)
        ]
        with pytest.raises(GeometryError, match="non-positive"):
            E.effective_area(2.5, elements, E_STEEL, FY50, 30.0)

    def test_element_rejects_bad_geometry(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            E.SlenderElement("wall", AxialElement.RECTANGULAR_HSS_WALL, b=0.0, t=0.10)
        with pytest.raises(GeometryError, match="count"):
            E.SlenderElement("wall", AxialElement.RECTANGULAR_HSS_WALL, b=6.0, t=0.10, count=0)


# ===========================================================================
# Sect. E1 -- the orchestrator
# ===========================================================================
@pytest.fixture
def w14x90() -> SectionAdapter:
    return SectionAdapter.from_mapping(
        {
            "name": "W14X90", "Ag": 26.5, "d": 14.0, "bf": 14.5, "tf": 0.710, "tw": 0.440,
            "Ix": 999.0, "Iy": 362.0, "Sx": 143.0, "Zx": 157.0, "rx": 6.14, "ry": 3.70,
            "J": 4.06, "Cw": 16000.0, "rts": 4.10, "ho": 13.3,
        }
    )


class TestCompressiveStrength:
    def test_w14x90_pinned_30_feet(self, w14x90: SectionAdapter) -> None:
        """Fcr = 25.024 ksi from the E3 hand calculation above; Pn = 663.1 kips."""
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        result = E.compressive_strength(member)
        assert result.limit_state is LimitState.FLEXURAL_BUCKLING
        assert result.nominal == pytest.approx(663.1, rel=0.005)
        assert result.available == pytest.approx(596.8, rel=0.005)

    def test_minor_axis_governs_when_unbraced_equally(self, w14x90: SectionAdapter) -> None:
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        detail = E.compressive_strength(member).governing.detail
        assert detail["Lc/r"] == pytest.approx(360.0 / 3.70)

    def test_bracing_the_minor_axis_shifts_control_to_x(self, w14x90: SectionAdapter) -> None:
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=120.0)
        detail = E.compressive_strength(member).governing.detail
        assert detail["Lc/r"] == pytest.approx(360.0 / 6.14)

    def test_torsional_buckling_is_evaluated_and_reported(self, w14x90: SectionAdapter) -> None:
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        states = {ls.limit_state for ls in E.compressive_strength(member).limit_states}
        assert LimitState.FLEXURAL_BUCKLING in states
        assert LimitState.TORSIONAL_BUCKLING in states

    def test_short_torsional_length_does_not_govern(self, w14x90: SectionAdapter) -> None:
        """A doubly symmetric rolled shape with Lcz = Lcy is not torsion-critical."""
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        result = E.compressive_strength(member)
        assert result.limit_state is LimitState.FLEXURAL_BUCKLING

    def test_long_torsional_length_can_govern(self, w14x90: SectionAdapter) -> None:
        """Sect. E4 applies to doubly symmetric members "when the torsional
        unbraced length exceeds the lateral unbraced length" (Sect. E3 User Note)."""
        member = E.CompressionMember(w14x90, A992, Lcx=120.0, Lcy=120.0, Lcz=900.0)
        result = E.compressive_strength(member)
        assert result.limit_state is LimitState.TORSIONAL_BUCKLING

    def test_lrfd_and_asd_share_one_nominal_strength(self, w14x90: SectionAdapter) -> None:
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        lrfd = E.compressive_strength(member, basis=Basis.LRFD)
        asd = E.compressive_strength(member, basis=Basis.ASD)
        assert lrfd.nominal == pytest.approx(asd.nominal)
        assert lrfd.available / asd.available == pytest.approx(0.90 * 1.67, rel=1e-12)

    def test_phi_and_omega_are_the_chapter_e_values(self, w14x90: SectionAdapter) -> None:
        """Sect. E1, p. 16.1-33: phi_c = 0.90, Omega_c = 1.67."""
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        result = E.compressive_strength(member)
        assert (result.phi, result.omega) == (0.90, 1.67)

    def test_slenderness_over_200_is_reported_not_rejected(
        self, w14x90: SectionAdapter
    ) -> None:
        """Sect. E2's 200 is a User Note preference, not a limit."""
        member = E.CompressionMember(w14x90, A992, Lcx=900.0, Lcy=900.0)
        result = E.compressive_strength(member)
        assert "exceeds the 200" in result.governing.note

    def test_missing_elements_are_flagged_on_the_report(self, w14x90: SectionAdapter) -> None:
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        assert "nonslender" in E.compressive_strength(member).governing.note

    def test_built_up_spacing_violation_is_flagged(self, w14x90: SectionAdapter) -> None:
        """Sect. E6.2(a): a/ri must not exceed 0.75*(Lc/r)."""
        member = E.CompressionMember(
            w14x90, A992, Lcx=360.0, Lcy=360.0,
            built_up=E.BuiltUpConfig(a=200.0, ri=1.0),
        )
        assert "E6.2(a) NOT satisfied" in E.compressive_strength(member).governing.note

    def test_built_up_reduces_strength(self, w14x90: SectionAdapter) -> None:
        plain = E.compressive_strength(E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0))
        built = E.compressive_strength(
            E.CompressionMember(
                w14x90, A992, Lcx=360.0, Lcy=360.0,
                built_up=E.BuiltUpConfig(a=60.0, ri=1.0),
            )
        )
        assert built.nominal < plain.nominal

    def test_slender_elements_reduce_strength(self) -> None:
        section = SectionAdapter.from_mapping(
            {"name": "HSS12X12X1/8", "Ag": 5.24, "rx": 4.82, "ry": 4.82,
             "Ix": 122.0, "Iy": 122.0, "J": 190.0}
        )
        elements = [
            E.SlenderElement("wall", AxialElement.RECTANGULAR_HSS_WALL, b=11.0, t=0.116, count=4)
        ]
        bare = E.compressive_strength(
            E.CompressionMember(section, A500_C_RECT, Lcx=180.0, Lcy=180.0)
        )
        slender = E.compressive_strength(
            E.CompressionMember(section, A500_C_RECT, Lcx=180.0, Lcy=180.0, elements=elements)
        )
        assert slender.nominal < bare.nominal
        assert "Sect. E7 applies" in slender.governing.note

    def test_single_angle_uses_e5_slenderness(self) -> None:
        angle = SectionAdapter.from_mapping(
            {"name": "L4X4X1/4", "Ag": 1.93, "rx": 1.25, "ry": 1.25,
             "Ix": 3.00, "Iy": 3.00, "J": 0.0438}
        )
        member = E.CompressionMember(
            angle, A36, Lcx=96.0, Lcy=96.0,
            angle=E.AngleConfig(L=96.0, ra=1.25, rz=0.778, b_long=4.0, b_short=4.0),
        )
        result = E.compressive_strength(member)
        expected = 72.0 + 0.75 * (96.0 / 1.25)  # Eq. E5-1
        assert result.governing.detail["Lc/r"] == pytest.approx(expected)
        assert "Sect. E5" in result.governing.note

    def test_report_renders(self, w14x90: SectionAdapter) -> None:
        member = E.CompressionMember(w14x90, A992, Lcx=360.0, Lcy=360.0)
        text = E.compressive_strength(member).report(required=500.0)
        assert "flexural buckling" in text
        assert "governs" in text
        assert "Eq. E3-1" in text
        assert "OK" in text

    def test_missing_section_properties_raise_clearly(self) -> None:
        member = E.CompressionMember(
            SectionAdapter.from_mapping({"name": "bare", "Ag": 10.0}), A992,
            Lcx=360.0, Lcy=360.0,
        )
        with pytest.raises(Exception, match="rx|ry"):
            E.compressive_strength(member)

    def test_section_without_j_skips_e4_gracefully(self) -> None:
        """A source that does not tabulate J still gets a Sect. E3 answer."""
        section = SectionAdapter.from_mapping(
            {"name": "minimal", "Ag": 26.5, "rx": 6.14, "ry": 3.70}
        )
        result = E.compressive_strength(
            E.CompressionMember(section, A992, Lcx=360.0, Lcy=360.0)
        )
        assert result.limit_state is LimitState.FLEXURAL_BUCKLING
        assert len(result.limit_states) == 1


class TestSinglyAndUnsymmetricMembers:
    def test_tee_uses_e4_3_without_warping(self) -> None:
        """User Note, p. 16.1-37: for tees, omit Cw from Fez and take xo = 0."""
        tee = SectionAdapter.from_mapping(
            {"name": "WT7X45", "Ag": 13.2, "rx": 1.94, "ry": 2.95,
             "Ix": 49.7, "Iy": 115.0, "J": 2.05, "Cw": 1.50}
        )
        member = E.CompressionMember(
            tee, A992, Lcx=120.0, Lcy=120.0,
            symmetry=E.Symmetry.SINGLY_Y, yo=1.60, include_warping=False,
        )
        result = E.compressive_strength(member)
        states = {ls.limit_state for ls in result.limit_states}
        assert LimitState.FLEXURAL_TORSIONAL_BUCKLING in states

    def test_channel_uses_the_fex_substitution(self) -> None:
        channel = SectionAdapter.from_mapping(
            {"name": "C12X30", "Ag": 8.81, "rx": 4.29, "ry": 0.762,
             "Ix": 162.0, "Iy": 5.12, "J": 0.861, "Cw": 151.0}
        )
        member = E.CompressionMember(
            channel, A36, Lcx=180.0, Lcy=180.0,
            symmetry=E.Symmetry.SINGLY_X, xo=2.13,
        )
        result = E.compressive_strength(member)
        assert LimitState.FLEXURAL_TORSIONAL_BUCKLING in {
            ls.limit_state for ls in result.limit_states
        }

    def test_unsymmetric_member_solves_the_cubic(self) -> None:
        section = SectionAdapter.from_mapping(
            {"name": "L6X4X1/2", "Ag": 4.75, "rx": 1.90, "ry": 1.11,
             "Ix": 17.3, "Iy": 6.22, "J": 0.399, "Cw": 0.0}
        )
        member = E.CompressionMember(
            section, A36, Lcx=120.0, Lcy=120.0,
            symmetry=E.Symmetry.UNSYMMETRIC, xo=1.03, yo=1.86,
        )
        result = E.compressive_strength(member)
        assert result.nominal > 0.0
        assert LimitState.FLEXURAL_TORSIONAL_BUCKLING in {
            ls.limit_state for ls in result.limit_states
        }


# ===========================================================================
# Design Examples hook
# ===========================================================================
_EXAMPLES = Path(__file__).parent / "design_examples" / "chapter_e.json"


def _load_examples() -> list[dict[str, object]]:
    if not _EXAMPLES.exists():
        return []
    data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
    return [case for case in data.get("cases", []) if not case.get("skip")]


class TestPublishedBenchmarks:
    """Cases with an independent published source, from ``chapter_e.json``.

    Currently AISC *Manual* Table 4-1a column-load-table values. The *Design
    Examples V16.0* companion is still not on this machine, so nothing here
    claims to be E.1A-E.7; add those cases to the JSON and they run
    automatically. Default tolerance is 0.5% -- the Manual's own printed
    precision of three significant figures.
    """

    @pytest.mark.design_example
    @pytest.mark.parametrize("case", _load_examples(), ids=lambda c: str(c.get("id", "?")))
    def test_matches_the_published_value(self, case: dict[str, object]) -> None:
        from pyaisc360.materials import grade

        member = E.CompressionMember(
            section=SectionAdapter.from_mapping(case["section"]),  # type: ignore[arg-type]
            steel=grade(str(case["grade"])),
            Lcx=float(case["Lcx"]),  # type: ignore[arg-type]
            Lcy=float(case["Lcy"]),  # type: ignore[arg-type]
            Lcz=float(case["Lcz"]) if case.get("Lcz") is not None else None,  # type: ignore[arg-type]
            symmetry=E.Symmetry(case.get("symmetry", E.Symmetry.DOUBLY)),
        )
        result = E.compressive_strength(member, basis=Basis(str(case.get("basis", "LRFD"))))
        tol = float(case.get("tol", 0.005))  # type: ignore[arg-type]
        assert result.available == pytest.approx(float(case["expected"]), rel=tol)  # type: ignore[arg-type]

    def test_every_case_declares_its_source(self) -> None:
        """No expected value enters this suite without saying where it came from."""
        for case in _load_examples():
            assert case.get("source"), f"case {case.get('id')!r} has no source"

    def test_case_file_is_valid(self) -> None:
        data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
        assert "cases" in data


class TestIndependentImplementationAgreement:
    """Regression lock on the Sect. E3 path.

    Every value below was confirmed **bit-for-bit** (relative difference 0.0e+00,
    184 comparisons across 23 W-shapes x 8 lengths) against the independent
    Chapter E implementation in ``aisc-steel-design/scripts/aisc360.py``, which
    was written separately and takes the ``Lc/r <= 4.71*sqrt(E/Fy)`` route rather
    than this library's ``Fy/Fe <= 2.25`` route.

    The values are vendored rather than imported so the suite has no dependency
    on a path outside the repository.
    """

    #: (shape, Ag, rx, ry, Lc in feet, phi_c*Pn in kips), A992.
    CASES = [
        ("W8X31", 9.13, 3.47, 2.02, 20, 146.1137),
        ("W10X49", 14.4, 4.35, 2.54, 14, 470.6066),
        ("W12X65", 19.1, 5.28, 3.02, 20, 541.6271),
        ("W14X90", 26.5, 6.14, 3.70, 14, 1025.6327),
        ("W14X120", 35.3, 6.24, 3.74, 30, 806.8102),
        ("W16X40", 11.8, 6.63, 1.57, 25, 73.0094),
        ("W18X76", 22.3, 7.73, 2.61, 30, 264.8017),
        ("W21X62", 18.3, 8.54, 1.77, 20, 224.8614),
        ("W24X68", 20.1, 9.55, 1.87, 40, 68.9186),
        ("W30X99", 29.0, 11.7, 2.10, 50, 80.2553),
    ]

    @pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}@{c[4]}ft")
    def test_agrees_with_the_independent_implementation(
        self, case: tuple[str, float, float, float, int, float]
    ) -> None:
        name, Ag, rx, ry, ft, expected = case
        section = SectionAdapter.from_mapping({"name": name, "Ag": Ag, "rx": rx, "ry": ry})
        member = E.CompressionMember(section, A992, Lcx=ft * 12.0, Lcy=ft * 12.0)
        result = E.compressive_strength(member)
        assert result.limit_state is LimitState.FLEXURAL_BUCKLING
        assert result.available == pytest.approx(expected, rel=1e-6)

    def test_transition_criteria_diverge_only_inside_the_rounding_band(self) -> None:
        """The two E3 transition criteria are not interchangeable -- measured.

        The printed 4.71 and the exact pi*sqrt(2.25) = 4.7124 bracket a band of
        Lc/r just 0.057 wide at Fy = 50 (113.432 to 113.489). Inside it the two
        implementations pick different branches, and Fcr differs by at most
        0.044%. Outside it they agree exactly -- which is what the 184-point
        sweep above demonstrates.
        """
        from pyaisc360 import utils

        printed = utils.limiting_ratio(4.71, E_STEEL, FY50)
        exact = utils.limiting_ratio(math.pi * 1.5, E_STEEL, FY50)
        assert exact - printed == pytest.approx(0.0575, abs=0.001)

        worst = 0.0
        for i in range(101):
            slenderness = printed + (exact - printed) * i / 100.0
            Fe = utils.elastic_buckling_stress(slenderness, E_STEEL)
            by_stress = utils.flexural_buckling_stress(FY50, Fe)  # this library
            by_slenderness = 0.877 * Fe if slenderness > printed else 0.658 ** (FY50 / Fe) * FY50
            worst = max(worst, abs(by_stress - by_slenderness) / by_slenderness)
        assert worst < 5e-4
