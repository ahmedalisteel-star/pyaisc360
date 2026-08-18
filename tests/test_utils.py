"""Chapter-independent mechanics.

Where the Specification publishes a value, the test asserts against *that*
rather than against a number this library produced -- the Cb cases below come
straight from the User Note to Sect. F1, p. 16.1-46.
"""

from __future__ import annotations

import math

import pytest

from pyaisc360 import utils
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import AxialSlenderness, FlexuralSlenderness
from pyaisc360.core.exceptions import AISC360Error, GeometryError

E = 29000.0


# ---------------------------------------------------------------------------
# Eq. B3-1 / B3-2
# ---------------------------------------------------------------------------
class TestAvailableStrength:
    def test_lrfd_applies_phi(self) -> None:
        assert utils.available_strength(100.0, phi=0.90, omega=1.67, basis=Basis.LRFD) == 90.0

    def test_asd_divides_by_omega(self) -> None:
        got = utils.available_strength(100.0, phi=0.90, omega=1.67, basis=Basis.ASD)
        assert got == pytest.approx(59.8802, abs=1e-4)

    @pytest.mark.parametrize(("phi", "omega"), [(0.90, 1.67), (0.75, 2.00), (1.00, 1.50)])
    def test_specification_factor_pairs_are_calibrated(self, phi: float, omega: float) -> None:
        """Every phi/Omega pair in the Specification satisfies Omega = 1.5/phi."""
        assert utils.available_strength(10.0, phi=phi, omega=omega, basis=Basis.LRFD) > 0.0

    def test_lrfd_and_asd_differ_by_the_calibration_factor(self) -> None:
        lrfd = utils.available_strength(100.0, phi=0.90, omega=1.67, basis=Basis.LRFD)
        asd = utils.available_strength(100.0, phi=0.90, omega=1.67, basis=Basis.ASD)
        assert lrfd / asd == pytest.approx(1.5, rel=2e-3)

    def test_transposed_factors_are_rejected(self) -> None:
        """phi=1.67/omega=0.90 has the right *product*, so only the bounds catch it."""
        with pytest.raises(AISC360Error, match="transposed"):
            utils.available_strength(100.0, phi=1.67, omega=0.90, basis=Basis.LRFD)

    def test_uncalibrated_pair_is_rejected(self) -> None:
        with pytest.raises(AISC360Error, match="not mutually calibrated"):
            utils.available_strength(100.0, phi=0.90, omega=1.20, basis=Basis.LRFD)

    def test_calibration_check_can_be_waived(self) -> None:
        got = utils.available_strength(
            100.0, phi=1.67, omega=0.90, basis=Basis.LRFD, check_calibration=False
        )
        assert got == 167.0

    def test_negative_nominal_strength_is_rejected(self) -> None:
        with pytest.raises(AISC360Error, match="non-negative"):
            utils.available_strength(-1.0, phi=0.90, omega=1.67, basis=Basis.LRFD)


# ---------------------------------------------------------------------------
# Eq. E3-4
# ---------------------------------------------------------------------------
class TestElasticBucklingStress:
    def test_matches_euler(self) -> None:
        assert utils.elastic_buckling_stress(100.0, E) == pytest.approx(
            math.pi**2 * E / 100.0**2, rel=1e-12
        )

    def test_slenderness_80(self) -> None:
        assert utils.elastic_buckling_stress(80.0, E) == pytest.approx(44.7216, abs=1e-4)

    def test_inverse_square(self) -> None:
        assert utils.elastic_buckling_stress(50.0, E) == pytest.approx(
            4.0 * utils.elastic_buckling_stress(100.0, E), rel=1e-12
        )

    @pytest.mark.parametrize("bad", [0.0, -10.0])
    def test_non_positive_slenderness_raises(self, bad: float) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            utils.elastic_buckling_stress(bad, E)


# ---------------------------------------------------------------------------
# Eq. E3-2 / E3-3
# ---------------------------------------------------------------------------
class TestFlexuralBucklingStress:
    def test_inelastic_branch(self) -> None:
        """Lc/r = 80, Fy = 50: Fy/Fe = 1.118 <= 2.25, so Eq. E3-2 governs."""
        Fe = utils.elastic_buckling_stress(80.0, E)
        assert utils.flexural_buckling_stress(50.0, Fe) == pytest.approx(31.3142, abs=1e-4)

    def test_elastic_branch(self) -> None:
        """Lc/r = 160, Fy = 50: Fy/Fe = 4.47 > 2.25, so Eq. E3-3 governs."""
        Fe = utils.elastic_buckling_stress(160.0, E)
        assert utils.flexural_buckling_stress(50.0, Fe) == pytest.approx(0.877 * Fe, rel=1e-12)

    def test_branches_agree_at_the_transition(self) -> None:
        """At Fy/Fe = 2.25 exactly, both equations give 0.877*Fe -- the curves meet."""
        Fy = 50.0
        Fe = Fy / 2.25
        assert utils.flexural_buckling_stress(Fy, Fe) == pytest.approx(0.877 * Fe, rel=1e-3)

    @pytest.mark.parametrize("Fy", [36.0, 50.0, 65.0, 70.0])
    def test_stress_and_slenderness_criteria_agree_exactly(self, Fy: float) -> None:
        """The Sect. E3 User Note, p. 16.1-36: the Fy/Fe <= 2.25 and
        Lc/r <= 4.71*sqrt(E/Fy) criteria give the same result.

        Exactly equivalent at the unrounded coefficient pi*sqrt(2.25) = 4.7124,
        since Lc/r = pi*sqrt(E/Fe).
        """
        exact = utils.limiting_ratio(math.pi * 1.5, E, Fy)
        for slenderness in (exact - 0.01, exact + 0.01):
            Fe = utils.elastic_buckling_stress(slenderness, E)
            assert (Fy / Fe <= 2.25) == (slenderness <= exact)

    @pytest.mark.parametrize("Fy", [36.0, 50.0, 65.0, 70.0])
    def test_printed_coefficient_471_is_the_rounded_form(self, Fy: float) -> None:
        """4.71 as printed differs from the exact pi*sqrt(2.25) by ~0.05%.

        Small enough to be irrelevant in design, but it means the two criteria
        are not bit-for-bit interchangeable -- which is why the library commits
        to the stress form everywhere.
        """
        printed = utils.limiting_ratio(4.71, E, Fy)
        exact = utils.limiting_ratio(math.pi * 1.5, E, Fy)
        assert abs(printed - exact) / exact < 6e-4

    def test_fcr_never_exceeds_fy(self) -> None:
        for slenderness in range(10, 300, 5):
            Fe = utils.elastic_buckling_stress(float(slenderness), E)
            assert utils.flexural_buckling_stress(50.0, Fe) <= 50.0

    def test_fcr_decreases_with_slenderness(self) -> None:
        stresses = [
            utils.flexural_buckling_stress(50.0, utils.elastic_buckling_stress(float(s), E))
            for s in range(20, 200, 10)
        ]
        assert all(a > b for a, b in zip(stresses, stresses[1:], strict=False))

    @pytest.mark.parametrize(("Fy", "Fe"), [(0.0, 30.0), (-50.0, 30.0), (50.0, 0.0), (50.0, -1.0)])
    def test_non_positive_stresses_raise(self, Fy: float, Fe: float) -> None:
        with pytest.raises(AISC360Error, match="must be positive"):
            utils.flexural_buckling_stress(Fy, Fe)


# ---------------------------------------------------------------------------
# Sect. B4.1
# ---------------------------------------------------------------------------
class TestClassification:
    def test_axial_nonslender_at_the_limit(self) -> None:
        assert utils.classify_axial_element(10.0, 10.0) is AxialSlenderness.NONSLENDER

    def test_axial_slender_above_the_limit(self) -> None:
        assert utils.classify_axial_element(10.01, 10.0) is AxialSlenderness.SLENDER

    @pytest.mark.parametrize(
        ("lam", "expected"),
        [
            (5.0, FlexuralSlenderness.COMPACT),
            (9.15, FlexuralSlenderness.COMPACT),
            (9.16, FlexuralSlenderness.NONCOMPACT),
            (24.0, FlexuralSlenderness.NONCOMPACT),
            (24.1, FlexuralSlenderness.SLENDER),
        ],
    )
    def test_flexural_bands(self, lam: float, expected: FlexuralSlenderness) -> None:
        """Table B4.1b case 10 flange limits at Fy = 50 ksi: lambda_p 9.15, lambda_r 24.0."""
        assert utils.classify_flexural_element(lam, 9.15, 24.0) is expected

    def test_w14x90_flange_is_noncompact_at_fy_50(self) -> None:
        """The User Note to Sect. F2, p. 16.1-47, names W14x90 as one of the few
        A6 shapes without compact flanges at Fy = 50 ksi. bf/2tf = 10.2."""
        lam_p = utils.limiting_ratio(0.38, E, 50.0)
        lam_r = utils.limiting_ratio(1.0, E, 50.0)
        assert utils.classify_flexural_element(10.2, lam_p, lam_r) is FlexuralSlenderness.NONCOMPACT

    def test_w18x50_flange_is_compact_at_fy_50(self) -> None:
        lam_p = utils.limiting_ratio(0.38, E, 50.0)
        assert utils.classify_flexural_element(6.57, lam_p, utils.limiting_ratio(1.0, E, 50.0)) is (
            FlexuralSlenderness.COMPACT
        )

    def test_transposed_limits_are_rejected(self) -> None:
        with pytest.raises(AISC360Error, match="transposed"):
            utils.classify_flexural_element(10.0, 24.0, 9.15)

    def test_governing_slenderness_takes_the_worst(self) -> None:
        assert (
            utils.governing_slenderness(
                FlexuralSlenderness.COMPACT,
                FlexuralSlenderness.SLENDER,
                FlexuralSlenderness.NONCOMPACT,
            )
            is FlexuralSlenderness.SLENDER
        )

    def test_governing_slenderness_all_compact(self) -> None:
        assert (
            utils.governing_slenderness(FlexuralSlenderness.COMPACT, FlexuralSlenderness.COMPACT)
            is FlexuralSlenderness.COMPACT
        )


# ---------------------------------------------------------------------------
# Eq. F1-1 -- validated against the published User Note values
# ---------------------------------------------------------------------------
class TestLateralTorsionalModificationFactor:
    def test_uniform_moment(self) -> None:
        """User Note, p. 16.1-46: equal end moments of opposite sign -> Cb = 1.0."""
        assert utils.lateral_torsional_modification_factor(100.0, 100.0, 100.0, 100.0) == (
            pytest.approx(1.0, abs=1e-12)
        )

    def test_reverse_curvature(self) -> None:
        """User Note: equal end moments of the same sign -> Cb = 2.27.

        Linear diagram from +M to -M gives |MA| = 0.5M, |MB| = 0, |MC| = 0.5M.
        """
        got = utils.lateral_torsional_modification_factor(100.0, 50.0, 0.0, 50.0)
        assert got == pytest.approx(2.2727, abs=1e-4)
        assert round(got, 2) == 2.27

    def test_one_end_moment_zero(self) -> None:
        """User Note: one end moment zero -> Cb = 1.67.

        Linear diagram from M to 0 gives 0.75M, 0.50M, 0.25M.
        """
        got = utils.lateral_torsional_modification_factor(100.0, 75.0, 50.0, 25.0)
        assert got == pytest.approx(1.6667, abs=1e-4)
        assert round(got, 2) == 1.67

    def test_uses_absolute_values(self) -> None:
        assert utils.lateral_torsional_modification_factor(
            -100.0, -50.0, 0.0, 50.0
        ) == pytest.approx(2.2727, abs=1e-4)

    def test_cantilever_is_unity(self) -> None:
        """Sect. F1(c), p. 16.1-46: Cb = 1.0 for a cantilever with warping
        prevented at the support and the free end unbraced."""
        assert (
            utils.lateral_torsional_modification_factor(100.0, 75.0, 50.0, 25.0, cantilever=True)
            == 1.0
        )

    def test_mmax_must_be_the_maximum(self) -> None:
        with pytest.raises(GeometryError, match="must be the maximum"):
            utils.lateral_torsional_modification_factor(50.0, 100.0, 60.0, 20.0)

    def test_zero_moment_raises(self) -> None:
        with pytest.raises(GeometryError, match="undefined"):
            utils.lateral_torsional_modification_factor(0.0, 0.0, 0.0, 0.0)

    def test_cb_is_never_below_unity_for_a_linear_diagram(self) -> None:
        for end_ratio in [i / 10 for i in range(-10, 11)]:
            MA = abs(1.0 + (end_ratio - 1.0) * 0.25)
            MB = abs(1.0 + (end_ratio - 1.0) * 0.50)
            MC = abs(1.0 + (end_ratio - 1.0) * 0.75)
            cb = utils.lateral_torsional_modification_factor(1.0, MA, MB, MC)
            assert cb >= 1.0


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
class TestLimitingRatio:
    def test_e3_compression_limit_at_fy_50(self) -> None:
        """4.71*sqrt(E/Fy) = 113.4 at Fy = 50 ksi -- the Sect. E3 transition."""
        assert utils.limiting_ratio(4.71, E, 50.0) == pytest.approx(113.4318, abs=1e-4)

    def test_f2_lp_coefficient_at_fy_50(self) -> None:
        """Eq. F2-5: Lp = 1.76*ry*sqrt(E/Fy); for W18x50, ry = 1.65 in."""
        assert 1.65 * utils.limiting_ratio(1.76, E, 50.0) == pytest.approx(69.9, abs=0.1)

    def test_b4_1b_flange_limits_at_fy_50(self) -> None:
        assert utils.limiting_ratio(0.38, E, 50.0) == pytest.approx(9.152, abs=1e-3)
        assert utils.limiting_ratio(1.0, E, 50.0) == pytest.approx(24.083, abs=1e-3)

    def test_b4_1b_web_limits_at_fy_50(self) -> None:
        assert utils.limiting_ratio(3.76, E, 50.0) == pytest.approx(90.553, abs=1e-3)
        assert utils.limiting_ratio(5.70, E, 50.0) == pytest.approx(137.274, abs=1e-3)

    @pytest.mark.parametrize(("E_", "Fy"), [(29000.0, 0.0), (0.0, 50.0), (29000.0, -1.0)])
    def test_non_positive_inputs_raise(self, E_: float, Fy: float) -> None:
        with pytest.raises(AISC360Error, match="must be positive"):
            utils.limiting_ratio(1.0, E_, Fy)


class TestLinearTransition:
    def test_returns_upper_at_lambda_p(self) -> None:
        assert utils.linear_transition(100.0, 60.0, 10.0, 10.0, 20.0) == 100.0

    def test_returns_lower_at_lambda_r(self) -> None:
        assert utils.linear_transition(100.0, 60.0, 20.0, 10.0, 20.0) == 60.0

    def test_interpolates_linearly(self) -> None:
        assert utils.linear_transition(100.0, 60.0, 15.0, 10.0, 20.0) == pytest.approx(80.0)

    def test_clamps_below_lambda_p(self) -> None:
        assert utils.linear_transition(100.0, 60.0, 5.0, 10.0, 20.0) == 100.0

    def test_clamps_above_lambda_r(self) -> None:
        assert utils.linear_transition(100.0, 60.0, 25.0, 10.0, 20.0) == 60.0

    def test_rejects_inverted_band(self) -> None:
        with pytest.raises(AISC360Error, match="must exceed"):
            utils.linear_transition(100.0, 60.0, 15.0, 20.0, 10.0)
