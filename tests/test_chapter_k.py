"""Chapter K -- HSS and Box-Section Connections.

The chapter is table-driven and its traps are almost all *substitutions*: using
the round-HSS ``Qf`` for a rectangular joint, the round interaction equation for
a rectangular one, or the Chapter J directional weld increase where Sect. K5
forbids it. The tests are built around telling those pairs apart.

Precision: 1e-12 throughout. Chapter K's coefficients are exact as printed --
there is no rounding to accommodate, unlike Eqs. E3-2 or F2-6.
"""

from __future__ import annotations

import math

import pytest

from pyaisc360 import chapter_j as J
from pyaisc360 import chapter_k as K
from pyaisc360.core.config import Basis
from pyaisc360.core.exceptions import AISC360Error, GeometryError

FY = 46.0
FYB = 46.0


# ===========================================================================
# Sect. K1 -- parameters
# ===========================================================================
class TestK1Parameters:
    def test_available_stress_lrfd_and_asd(self) -> None:
        """Sect. K1.1: Fc = Fy for LRFD, 0.60*Fy for ASD.

        Note the direction: elsewhere an ASD *demand* is scaled up by 1.6; here
        the *capacity* is scaled down by 0.60 = 1/1.67.
        """
        assert K.available_stress(FY, Basis.LRFD) == FY
        assert K.available_stress(FY, Basis.ASD) == pytest.approx(0.60 * FY, rel=1e-12)

    def test_effective_width_falls_with_chord_slenderness(self) -> None:
        """Eq. K1-1: the 10/(B/t) term. A stocky chord is fully effective; a
        slender one transfers only part of the branch width."""
        stocky = K.effective_width(8.0, 0.291, FY, FYB, 0.5, 6.0)  # B/t = 27.5
        slender = K.effective_width(8.0, 0.174, FY, FYB, 0.5, 6.0)  # B/t = 46
        assert slender < stocky

    def test_effective_width_is_capped_at_the_branch_width(self) -> None:
        """A stocky chord would otherwise compute Be wider than the branch."""
        assert K.effective_width(8.0, 0.581, FY, FYB, 0.25, 6.0) == 6.0

    def test_effective_width_at_B_over_t_of_10_is_the_full_width(self) -> None:
        """10/(B/t) = 1 at B/t = 10, and equal yield/thickness gives Be = Bb."""
        assert K.effective_width(10.0, 1.0, FY, FY, 1.0, 6.0) == pytest.approx(6.0, rel=1e-12)

    def test_a_thicker_branch_reduces_its_own_effective_width(self) -> None:
        """Be goes as t/tb -- a stiff branch on a flexible chord concentrates
        its load at the side walls."""
        thin = K.effective_width(8.0, 0.291, FY, FYB, 0.25, 6.0)
        thick = K.effective_width(8.0, 0.291, FY, FYB, 0.75, 6.0)
        assert thick < thin


class TestQfParameters:
    def test_utilization_U(self) -> None:
        """Eq. K2-4: U = Pro/(Fc*Ag) + Mro/(Fc*S)."""
        assert K.utilization_U(200.0, 400.0, 46.0, 12.0, 20.0) == pytest.approx(
            200.0 / (46.0 * 12.0) + 400.0 / (46.0 * 20.0), rel=1e-12
        )

    def test_Qf_is_unity_when_the_chord_face_is_in_tension(self) -> None:
        """Eq. K2-3. A chord face in tension is stiffened by it, not softened --
        the asymmetry is real and often missed."""
        assert K.Qf(0.8, chord_in_tension=True) == 1.0
        assert K.Qf(0.8) < 1.0

    def test_Qf_compression_form(self) -> None:
        """Qf = 1.0 - 0.3*U*(1 + U)."""
        assert K.Qf(0.5) == pytest.approx(1.0 - 0.3 * 0.5 * 1.5, rel=1e-12)

    def test_Qf_decreases_with_utilisation(self) -> None:
        values = [K.Qf(u / 10.0) for u in range(0, 11)]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))

    def test_Qf_is_floored_at_zero(self) -> None:
        """The parabola crosses zero near U = 1.43; beyond that a negative Qf
        would flip the sign of every strength that multiplies it."""
        assert K.Qf(2.0) == 0.0
        assert K.Qf(5.0) == 0.0

    def test_rectangular_Qf_is_a_different_expression(self) -> None:
        """Eqs. K3-14/K3-15: Qf = 1.3 - 0.4*U/beta, capped at 1.0.

        Substituting the round-HSS Eq. K2-3 for it is a substitution the tables
        do not permit, and the two disagree materially at low beta.
        """
        U, beta = 0.6, 0.5
        assert K.Qf_rectangular(U, beta) == pytest.approx(1.3 - 0.4 * U / beta, rel=1e-12)
        assert K.Qf_rectangular(U, beta) != pytest.approx(K.Qf(U), rel=1e-3)

    def test_rectangular_Qf_penalises_a_narrow_branch(self) -> None:
        """Dividing U by beta means a narrow branch on a loaded chord is hit
        much harder -- the opposite of the round-HSS form, which ignores beta."""
        wide = K.Qf_rectangular(0.6, 0.9)
        narrow = K.Qf_rectangular(0.6, 0.3)
        assert narrow < wide

    def test_rectangular_Qf_is_capped_at_unity(self) -> None:
        assert K.Qf_rectangular(0.1, 0.9) == 1.0

    def test_rectangular_Qf_is_unity_in_tension(self) -> None:
        assert K.Qf_rectangular(0.9, 0.3, chord_in_tension=True) == 1.0


# ===========================================================================
# Sect. K2 -- concentrated forces
# ===========================================================================
class TestK2:
    def test_transverse_plate_axial_and_moment(self) -> None:
        """Eqs. K2-1a and K2-1b."""
        result = K.transverse_plate_to_round_hss(FY, 0.349, 10.0, 5.0, 90.0, 1.0)
        expected = FY * 0.349**2 * (5.5 / (1.0 - 0.81 * 0.5))
        assert result.nominal == pytest.approx(expected, rel=1e-12)
        assert result.governing.detail["Mn (Eq. K2-1b)"] == pytest.approx(
            0.5 * 5.0 * result.nominal, rel=1e-12
        )

    def test_transverse_plate_singularity_is_guarded(self) -> None:
        """1 - 0.81*beta hits zero at beta = 1.235. Table K2.1A caps beta at
        1.0, so the equation should never be evaluated past it -- but the guard
        catches a caller who ignores the limits table."""
        with pytest.raises(GeometryError, match="singular"):
            K.transverse_plate_to_round_hss(FY, 0.349, 10.0, 13.0, 90.0, 1.0)

    def test_longitudinal_plate_gains_with_bearing_length(self) -> None:
        """Eq. K2-2a's (1 + 0.25*lb/D) is the OPPOSITE sign to Eq. K2-1a's
        1/(1 - 0.81*Bb/D) -- a longitudinal plate loads the stiff meridian."""
        short = K.longitudinal_plate_to_round_hss(FY, 0.349, 10.0, 2.0, 90.0, 1.0)
        long_ = K.longitudinal_plate_to_round_hss(FY, 0.349, 10.0, 10.0, 90.0, 1.0)
        assert long_.nominal > short.nominal

    def test_longitudinal_moment_coefficient_differs(self) -> None:
        """Eq. K2-2b uses 0.8*lb where Eq. K2-1b uses 0.5*Bb."""
        result = K.longitudinal_plate_to_round_hss(FY, 0.349, 10.0, 6.0, 90.0, 1.0)
        assert result.governing.detail["Mn (Eq. K2-2b)"] == pytest.approx(
            0.8 * 6.0 * result.nominal, rel=1e-12
        )

    def test_K2_2b_is_the_recovered_equation(self) -> None:
        """The dump misprints Eq. K2-2b's label as (K2-2a); the scanner's
        EXTRACTION_CORRECTIONS restores it with the evidence recorded."""
        from pyaisc360.core.citations import cite, spec_index

        assert cite("K2-2b").section == "K2"
        corrected = {c["actually"] for c in spec_index()["corrections"]}
        assert "K2-2b" in corrected

    def test_inclined_plate_needs_more_strength(self) -> None:
        """Every Table K2.1 row divides by sin(theta)."""
        normal = K.longitudinal_plate_to_round_hss(FY, 0.349, 10.0, 6.0, 90.0, 1.0)
        inclined = K.longitudinal_plate_to_round_hss(FY, 0.349, 10.0, 6.0, 30.0, 1.0)
        assert inclined.nominal == pytest.approx(2.0 * normal.nominal, rel=1e-12)

    def test_zero_angle_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="30 degrees"):
            K.longitudinal_plate_to_round_hss(FY, 0.349, 10.0, 6.0, 0.0, 1.0)

    def test_plastification_factors(self) -> None:
        result = K.transverse_plate_to_round_hss(FY, 0.349, 10.0, 5.0, 90.0, 1.0)
        assert (result.phi, result.omega) == (0.90, 1.67)


# ===========================================================================
# Sect. K3 -- truss connections
# ===========================================================================
class TestK3Round:
    def test_punching_shear(self) -> None:
        """Eq. K3-1, at phi = 0.95."""
        result = K.round_punching_shear(FY, 0.349, 6.0, 90.0)
        assert (result.phi, result.omega) == (0.95, 1.58)

    def test_T_Y_plastification_grows_with_beta_squared(self) -> None:
        """Eq. K3-2: (3.1 + 15.6*beta^2)."""
        a = K.round_T_Y_plastification(FY, 0.349, 0.4, 14.3, 90.0, 1.0).nominal
        b = K.round_T_Y_plastification(FY, 0.349, 0.8, 14.3, 90.0, 1.0).nominal
        assert b / a == pytest.approx(
            (3.1 + 15.6 * 0.64) / (3.1 + 15.6 * 0.16), rel=1e-12
        )

    def test_gamma_enters_as_a_positive_power(self) -> None:
        """Eq. K3-2's gamma^0.2 makes a MORE slender chord compute stronger by
        that term alone -- the Fy*t^2 prefactor more than offsets it, but the
        sign of the exponent is genuinely positive and worth pinning."""
        thin = K.round_T_Y_plastification(FY, 0.349, 0.5, 30.0, 90.0, 1.0).nominal
        thick = K.round_T_Y_plastification(FY, 0.349, 0.5, 10.0, 90.0, 1.0).nominal
        assert thin > thick  # same t, larger gamma only

    def test_cross_connection_has_no_gamma_term(self) -> None:
        """Eq. K3-3 depends on beta alone -- a cross-connection cannot develop
        the ring action a T-connection relies on."""
        result = K.round_cross_plastification(FY, 0.349, 0.5, 90.0, 1.0)
        expected = FY * 0.349**2 * (5.7 / (1.0 - 0.81 * 0.5))
        assert result.nominal == pytest.approx(expected, rel=1e-12)

    def test_cross_connection_singularity_is_guarded(self) -> None:
        with pytest.raises(GeometryError, match="singular"):
            K.round_cross_plastification(FY, 0.349, 1.3, 90.0, 1.0)

    def test_K_connection_tension_branch_takes_the_compression_value(self) -> None:
        """Eq. K3-5 is the one people drop: the tension branch is NOT evaluated
        independently -- it takes the compression branch's value, computed from
        the COMPRESSION branch's diameter."""
        result = K.round_K_plastification(FY, 0.349, 5.0, 10.0, 45.0, 1.2, 1.0)
        assert "TENSION branch" in result.governing.note
        assert "COMPRESSION branch" in result.governing.note


class TestQg:
    """Eq. K3-6 -- the chapter's only exponential, and its overflow trap."""

    def test_a_small_gap_gives_the_maximum(self) -> None:
        """exp(0.5*g/t - 1.33) -> 0 as the gap closes, so the bracket tends to
        1 + 0.024*gamma^1.2. Closely spaced branches share chord material."""
        gamma = 14.3
        tight = K.Qg(gamma, 0.0, 0.349)
        assert tight == pytest.approx(
            gamma**0.2 * (1.0 + 0.024 * gamma**1.2 / (math.exp(-1.33) + 1.0)), rel=1e-12
        )

    def test_a_large_gap_tends_to_gamma_to_the_0_2(self) -> None:
        """exp() -> inf drives the bracket to 1."""
        gamma = 14.3
        assert K.Qg(gamma, 100.0, 0.349) == pytest.approx(gamma**0.2, rel=1e-6)

    def test_Qg_decreases_monotonically_with_the_gap(self) -> None:
        values = [K.Qg(14.3, g / 4.0, 0.349) for g in range(0, 40)]
        assert all(a >= b for a, b in zip(values, values[1:], strict=False))

    def test_an_enormous_gap_does_not_overflow(self) -> None:
        """0.5*g/t above ~710 overflows math.exp and would give inf, then nan.
        The large-gap limit is taken directly instead."""
        got = K.Qg(14.3, 1.0e6, 0.001)
        assert math.isfinite(got)
        assert got == pytest.approx(14.3**0.2, rel=1e-12)

    def test_never_below_the_large_gap_limit(self) -> None:
        gamma = 14.3
        for g in (0.0, 1.0, 10.0, 1e3, 1e9):
            assert K.Qg(gamma, g, 0.349) >= gamma**0.2 - 1e-12

    def test_negative_gap_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="non-negative"):
            K.Qg(14.3, -1.0, 0.349)


class TestK3Rectangular:
    def test_beta_eff_uses_both_branches(self) -> None:
        """Eq. K3-16: a K-joint's effective width ratio depends on the PAIR."""
        got = K.beta_eff(4.0, 4.0, 3.0, 3.0, 8.0)
        assert got == pytest.approx((8.0 + 6.0) / 32.0, rel=1e-12)

    def test_beta_eop_is_capped_at_beta(self) -> None:
        """Eq. K3-17: beta_eop = 5*beta/gamma <= beta, so a stocky chord
        (gamma < 5) does not get a bonus."""
        assert K.beta_eop(0.6, 4.0) == pytest.approx(0.6, rel=1e-12)
        assert K.beta_eop(0.6, 20.0) == pytest.approx(0.15, rel=1e-12)

    def test_gapped_K_plastification(self) -> None:
        """Eq. K3-7: applies "for all beta" -- no width-ratio branching."""
        result = K.rectangular_gapped_K_plastification(FY, 0.291, 0.44, 17.2, 45.0, 1.0)
        expected = FY * 0.291**2 * (9.8 * 0.44 * math.sqrt(17.2)) / math.sin(math.radians(45.0))
        assert result.nominal == pytest.approx(expected, rel=1e-12)

    def test_branch_local_yielding_removes_the_corners(self) -> None:
        """Eq. K3-9's -4*tb removes corners counted twice by the perimeter sum."""
        result = K.branch_local_yielding(FYB, 0.233, 4.0, 4.0, 3.2)
        assert result.nominal == pytest.approx(
            FYB * 0.233 * (2 * 4.0 + 4.0 + 3.2 - 4 * 0.233), rel=1e-12
        )

    def test_branch_local_yielding_rejects_chord_dimensions(self) -> None:
        """Passing chord dimensions makes the perimeter non-positive."""
        with pytest.raises(GeometryError, match="non-positive"):
            K.branch_local_yielding(FYB, 4.0, 1.0, 1.0, 1.0)

    def test_punching_factors(self) -> None:
        result = K.rectangular_punching_shear(FY, 0.291, 8.0, 0.5, 0.5, 0.15, 45.0)
        assert (result.phi, result.omega) == (0.95, 1.58)


class TestOverlappedK:
    ARGS = dict(Fybi=FYB, tbi=0.233, Hbi=4.0, Bbi=4.0, Bei=3.2, Bej=3.0)

    def test_the_three_overlap_bands(self) -> None:
        """Eqs. K3-10, K3-11, K3-12 at 25-50%, 50-80% and 80-100%."""
        for Ov, equation in ((30.0, "K3-10"), (60.0, "K3-11"), (90.0, "K3-12")):
            result = K.overlapped_K_branch_yielding(**self.ARGS, Ov=Ov)
            assert result.citation.equation == equation

    def test_below_50_percent_the_web_term_is_prorated(self) -> None:
        """Eq. K3-10 scales the web contribution by Ov/50."""
        at_50 = K.overlapped_K_branch_yielding(**self.ARGS, Ov=50.0).nominal
        at_25 = K.overlapped_K_branch_yielding(**self.ARGS, Ov=25.0).nominal
        web = 2 * 4.0 - 4 * 0.233
        assert at_50 - at_25 == pytest.approx(FYB * 0.233 * 0.5 * web, rel=1e-12)

    def test_the_bands_are_continuous_at_50(self) -> None:
        """Ov/50 = 1 at Ov = 50, so Eqs. K3-10 and K3-11 meet exactly."""
        below = K.overlapped_K_branch_yielding(**self.ARGS, Ov=49.99999).nominal
        at = K.overlapped_K_branch_yielding(**self.ARGS, Ov=50.0).nominal
        assert below == pytest.approx(at, rel=1e-5)

    def test_at_80_percent_the_full_branch_width_replaces_the_effective_one(self) -> None:
        """Eq. K3-12 swaps Bei for Bbi -- at that overlap the branch bears on the
        other branch, not on the flexible chord face. A genuine step up."""
        just_below = K.overlapped_K_branch_yielding(**self.ARGS, Ov=79.999).nominal
        at_80 = K.overlapped_K_branch_yielding(**self.ARGS, Ov=80.0).nominal
        assert at_80 > just_below
        assert at_80 - just_below == pytest.approx(
            FYB * 0.233 * (4.0 - 3.2), rel=1e-4
        )

    def test_overlap_below_25_percent_is_rejected(self) -> None:
        """Table K3.2A's limits start at 25%; Ov is a percentage, not a fraction."""
        with pytest.raises(AISC360Error, match="percentage, not a fraction"):
            K.overlapped_K_branch_yielding(**self.ARGS, Ov=0.5)

    def test_the_overlapped_branch_scales_from_the_overlapping_one(self) -> None:
        """Eq. K3-13: Pn,j = Pn,i*(Fybj*Abj)/(Fybi*Abi). Subscript i is the
        OVERLAPPING branch; swapping them inverts the ratio."""
        Pn_i = 100.0
        assert K.overlapped_K_other_branch(Pn_i, 46.0, 3.0, 46.0, 6.0) == pytest.approx(
            200.0, rel=1e-12
        )
        assert K.overlapped_K_other_branch(Pn_i, 46.0, 6.0, 46.0, 3.0) == pytest.approx(
            50.0, rel=1e-12
        )


# ===========================================================================
# Sect. K4 -- moment connections
# ===========================================================================
class TestK4:
    def test_in_plane_uses_gamma_to_the_half(self) -> None:
        """Eq. K4-1's gamma^0.5 against Eq. K3-2's gamma^0.2 -- bending mobilises
        chord slenderness more strongly than axial load."""
        result = K.round_ip_plastification(FY, 0.349, 6.0, 0.6, 14.3, 90.0, 1.0)
        expected = 5.39 * FY * 0.349**2 * math.sqrt(14.3) * 0.6 * 6.0
        assert result.nominal == pytest.approx(expected, rel=1e-12)

    def test_the_punching_numerators_swap(self) -> None:
        """Eq. K4-2 has (1 + 3*sin) and Eq. K4-4 has (3 + sin). They coincide at
        theta = 90 (both 4) and diverge as the angle drops."""
        ip90 = K.round_ip_punching(FY, 0.349, 6.0, 90.0).nominal
        op90 = K.round_op_punching(FY, 0.349, 6.0, 90.0).nominal
        assert ip90 == pytest.approx(op90, rel=1e-12)
        ip30 = K.round_ip_punching(FY, 0.349, 6.0, 30.0).nominal
        op30 = K.round_op_punching(FY, 0.349, 6.0, 30.0).nominal
        assert op30 > ip30

    def test_out_of_plane_plastification_singularity(self) -> None:
        with pytest.raises(GeometryError, match="singular"):
            K.round_op_plastification(FY, 0.349, 6.0, 1.3, 90.0, 1.0)

    def test_chord_distortional_has_phi_one(self) -> None:
        """Eq. K4-7 is the chapter's only phi = 1.00 -- and the fifth in the
        Specification, alongside Sects. G2.1(a), J3.8, J4.2(a) and J10.2."""
        result = K.rectangular_chord_distortional(FY, 0.291, 4.0, 8.0, 8.0)
        assert (result.phi, result.omega) == (1.00, 1.50)
        assert result.available == pytest.approx(result.nominal, rel=1e-12)

    def test_chord_distortional_value(self) -> None:
        result = K.rectangular_chord_distortional(FY, 0.291, 4.0, 8.0, 8.0)
        expected = 2 * FY * 0.291 * (4.0 * 0.291 + math.sqrt(8.0 * 8.0 * 0.291 * 16.0))
        assert result.nominal == pytest.approx(expected, rel=1e-12)

    def test_chord_distortional_applies_only_to_unbalanced_joints(self) -> None:
        note = K.rectangular_chord_distortional(FY, 0.291, 4.0, 8.0, 8.0).governing.note
        assert "UNBALANCED" in note


class TestMomentInteraction:
    """The single most consequential difference in the chapter."""

    def test_round_squares_the_in_plane_term_only(self) -> None:
        """Eq. K4-5: Pr/Pc + (Mr-ip/Mc-ip)^2 + Mr-op/Mc-op."""
        result = K.round_moment_interaction(30.0, 100.0, 40.0, 100.0, 20.0, 100.0)
        assert result.ratio == pytest.approx(0.3 + 0.4**2 + 0.2, rel=1e-12)

    def test_rectangular_is_fully_linear(self) -> None:
        """Eq. K4-8: no squared term at all."""
        result = K.rectangular_moment_interaction(30.0, 100.0, 40.0, 100.0, 20.0, 100.0)
        assert result.ratio == pytest.approx(0.3 + 0.4 + 0.2, rel=1e-12)

    def test_using_the_round_form_for_a_rectangular_joint_is_unconservative(self) -> None:
        """Squaring a ratio below 1.0 reduces it, so the round form understates a
        rectangular joint wherever in-plane bending is significant."""
        args = (30.0, 100.0, 40.0, 100.0, 20.0, 100.0)
        assert K.round_moment_interaction(*args).ratio < (
            K.rectangular_moment_interaction(*args).ratio
        )

    def test_they_agree_only_when_in_plane_bending_is_absent(self) -> None:
        args = (30.0, 100.0, 0.0, 100.0, 20.0, 100.0)
        assert K.round_moment_interaction(*args).ratio == pytest.approx(
            K.rectangular_moment_interaction(*args).ratio, rel=1e-12
        )

    def test_asd_selects_the_other_equation_label(self) -> None:
        args = (30.0, 100.0, 40.0, 100.0, 20.0, 100.0)
        assert K.round_moment_interaction(*args, Basis.ASD).citation.equation == "K4-6"
        assert K.rectangular_moment_interaction(
            *args, Basis.ASD
        ).citation.equation == "K4-9"

    def test_adequacy(self) -> None:
        ok = K.rectangular_moment_interaction(10.0, 100.0, 10.0, 100.0, 10.0, 100.0)
        assert ok.is_adequate
        over = K.rectangular_moment_interaction(50.0, 100.0, 50.0, 100.0, 50.0, 100.0)
        assert not over.is_adequate


# ===========================================================================
# Sect. K5 -- welds
# ===========================================================================
class TestK5Welds:
    def test_no_directional_strength_increase(self) -> None:
        """Sect. K5: Fnw carries "no increase in strength due to directionality
        of load for fillet welds". Eq. J2-5's bonus is NOT available -- le
        already accounts for the non-uniform load transfer, so applying both
        double-counts."""
        result = K.weld_axial_strength(0.60 * 70.0, 0.177, 20.0)
        assert "NO directional" in result.governing.note
        # Chapter J's transverse increase would have been 1.5x.
        assert J.directional_strength_increase(90.0) == pytest.approx(1.5, rel=1e-12)

    def test_fillet_and_pjp_factors(self) -> None:
        fillet = K.weld_axial_strength(42.0, 0.177, 20.0)
        pjp = K.weld_axial_strength(42.0, 0.177, 20.0, pjp=True)
        assert (fillet.phi, fillet.omega) == (0.75, 2.00)
        assert (pjp.phi, pjp.omega) == (0.80, 1.88)
        assert pjp.available > fillet.available

    def test_transverse_plate_length_counts_both_sides(self) -> None:
        """Eq. K5-4: le = 2*Be, the two sides of the plate."""
        assert K.transverse_plate_weld_length(3.2) == pytest.approx(6.4, rel=1e-12)

    def test_branch_length_divides_only_the_webs_by_sin(self) -> None:
        """Eq. K5-5: 2*Hb/sin(theta) + 2*Be. The flanges are not inclined."""
        got = K.branch_weld_length(4.0, 3.2, 30.0)
        assert got == pytest.approx(2 * 4.0 / 0.5 + 2 * 3.2, rel=1e-12)

    def test_section_moduli(self) -> None:
        """Eqs. K5-6 and K5-7."""
        Sip = K.branch_weld_Sip(0.177, 4.0, 3.2, 90.0)
        assert Sip == pytest.approx(0.177 / 3 * 16.0 + 0.177 * 3.2 * 4.0, rel=1e-12)
        Sop = K.branch_weld_Sop(0.177, 4.0, 4.0, 3.2, 90.0)
        expected = (
            0.177 * 4.0 * 4.0 + 0.177 / 3 * 16.0 - 0.177 / 3 * ((4.0 - 3.2) ** 3 / 4.0)
        )
        assert Sop == pytest.approx(expected, rel=1e-12)

    def test_Sop_rejects_Be_above_Bb(self) -> None:
        with pytest.raises(GeometryError, match="cannot exceed"):
            K.branch_weld_Sop(0.177, 4.0, 4.0, 5.0, 90.0)


class TestGappedKWeldInterpolation:
    """Eqs. K5-8/K5-9 and the 50-60 degree interpolation Table K5.1 mandates."""

    ARGS = dict(Hb=4.0, Bb=4.0, tb=0.233)

    def test_below_50_degrees(self) -> None:
        le, note = K.gapped_K_weld_length(**self.ARGS, theta=45.0)
        assert "K5-8" in note

    def test_at_or_above_60_degrees(self) -> None:
        le, note = K.gapped_K_weld_length(**self.ARGS, theta=60.0)
        assert "K5-9" in note

    def test_the_two_expressions_differ_by_a_whole_flange_term(self) -> None:
        """Eq. K5-8 has 2*(Bb - 1.2tb) where Eq. K5-9 has one -- so treating the
        boundary as a step overstates a 55-degree joint by up to a third."""
        at_50, _ = K.gapped_K_weld_length(**self.ARGS, theta=50.0)
        at_60, _ = K.gapped_K_weld_length(**self.ARGS, theta=60.0)
        # both evaluated at their own sin(theta), so compare the flange gap only
        flange = 4.0 - 1.2 * 0.233
        short_at_60 = 2 * (4.0 - 1.2 * 0.233) / math.sin(math.radians(60.0)) + 2 * flange
        assert short_at_60 - at_60 == pytest.approx(flange, rel=1e-12)

    def test_interpolation_is_applied_between_50_and_60(self) -> None:
        le, note = K.gapped_K_weld_length(**self.ARGS, theta=55.0)
        assert "interpolation" in note
        s = math.sin(math.radians(55.0))
        web, flange = 4.0 - 1.2 * 0.233, 4.0 - 1.2 * 0.233
        short = 2 * web / s + 2 * flange
        long_ = 2 * web / s + flange
        assert le == pytest.approx(short + (long_ - short) * 0.5, rel=1e-12)

    def test_interpolation_is_continuous_at_both_ends(self) -> None:
        below, _ = K.gapped_K_weld_length(**self.ARGS, theta=50.0)
        just_above, _ = K.gapped_K_weld_length(**self.ARGS, theta=50.0001)
        assert below == pytest.approx(just_above, rel=1e-4)
        near_60, _ = K.gapped_K_weld_length(**self.ARGS, theta=59.9999)
        at_60, _ = K.gapped_K_weld_length(**self.ARGS, theta=60.0)
        assert near_60 == pytest.approx(at_60, rel=1e-4)

    def test_an_oversized_corner_deduction_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="corner deduction"):
            K.gapped_K_weld_length(Hb=0.2, Bb=0.2, tb=0.233, theta=45.0)


class TestOverlappedWeldLengths:
    ARGS = dict(Hbi=4.0, Bbi=4.0, Bei=3.2, Bej=3.0, theta_i=45.0, theta_j=45.0)

    def test_the_three_bands(self) -> None:
        for Ov, equation in ((30.0, "K5-10"), (60.0, "K5-11"), (90.0, "K5-12")):
            _, note = K.overlapping_branch_weld_length(**self.ARGS, Ov=Ov)
            assert equation in note

    def test_the_overlapped_portion_uses_the_combined_angle(self) -> None:
        """sin(theta_i + theta_j) -- that length of weld runs along the OTHER
        branch, not along the chord."""
        shallow = K.overlapping_branch_weld_length(
            **{**self.ARGS, "theta_j": 20.0}, Ov=60.0
        )[0]
        steep = K.overlapping_branch_weld_length(
            **{**self.ARGS, "theta_j": 45.0}, Ov=60.0
        )[0]
        assert shallow != pytest.approx(steep, rel=1e-6)

    def test_collinear_branches_are_rejected(self) -> None:
        with pytest.raises(GeometryError, match="collinear"):
            K.overlapping_branch_weld_length(
                Hbi=4.0, Bbi=4.0, Bei=3.2, Bej=3.0, theta_i=90.0, theta_j=90.0, Ov=60.0
            )

    def test_overlapped_branch_default(self) -> None:
        """Eq. K5-13: le,j = 2*Hbj/sin(theta_j) + 2*Bej."""
        le, note = K.overlapped_branch_weld_length(4.0, 3.0, 0.233, 30.0)
        assert le == pytest.approx(2 * 4.0 / 0.5 + 2 * 3.0, rel=1e-12)
        assert "K5-13" in note

    def test_wide_or_steep_drops_the_flange_entirely(self) -> None:
        """Eq. K5-14 removes the flange contribution -- not reduces it. The
        overlapping branch covers it."""
        le, note = K.overlapped_branch_weld_length(
            4.0, 3.0, 0.233, 60.0, wide_or_steep=True
        )
        s = math.sin(math.radians(60.0))
        assert le == pytest.approx(2 * (4.0 - 1.2 * 0.233) / s, rel=1e-12)
        assert "webs only" in note
        assert le < K.overlapped_branch_weld_length(4.0, 3.0, 0.233, 60.0)[0]


# ===========================================================================
# Sect. K1's "not prohibited" rule
# ===========================================================================
class TestApplicabilityReporting:
    def test_a_violation_reports_rather_than_raises(self) -> None:
        """Sect. K1, p. 16.1-149: connections outside the limits "are NOT
        prohibited and must be designed by rational analysis". Raising would be
        wrong -- the joint is permitted, just not by these equations."""
        check = K.ApplicabilityCheck(
            within_limits=False, table="Table K3.1A",
            violations=("D/t = 62 exceeds 50 for T-connections",),
        )
        assert not check.within_limits
        assert "rational analysis" in check.note
        assert "D/t = 62" in check.note
        assert "Table K3.1A" in check.note

    def test_a_compliant_connection_says_so(self) -> None:
        check = K.ApplicabilityCheck(within_limits=True, table="Table K3.1A")
        assert "within the limits" in check.note


class TestFactorCalibration:
    @pytest.mark.parametrize(
        ("phi", "omega", "where"),
        [
            (K.PHI_PLASTIFICATION, K.OMEGA_PLASTIFICATION, "chord plastification"),
            (K.PHI_PUNCHING, K.OMEGA_PUNCHING, "punching shear"),
            (K.PHI_DISTORTIONAL, K.OMEGA_DISTORTIONAL, "chord distortional"),
            (K.PHI_FILLET_WELD, K.OMEGA_FILLET_WELD, "fillet weld"),
            (K.PHI_PJP_WELD, K.OMEGA_PJP_WELD, "PJP weld"),
        ],
    )
    def test_every_pair_is_calibrated(self, phi: float, omega: float, where: str) -> None:
        """Omega = 1.5/phi, including the unusual 0.95/1.58."""
        assert phi * omega == pytest.approx(1.5, abs=0.01), where

    def test_the_0_95_pair_is_unique_to_chapter_K(self) -> None:
        """phi = 0.95 / Omega = 1.58 appears nowhere else in the Specification."""
        assert (K.PHI_PUNCHING, K.OMEGA_PUNCHING) == (0.95, 1.58)
        assert pytest.approx(1.501, abs=0.001) == 0.95 * 1.58
