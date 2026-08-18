"""Chapter D -- Design of Members for Tension.

The chapter's defining feature is that its two limit states carry **different**
resistance factors, so the governing state must be chosen on available strength.
:class:`TestOrderingInversion` demonstrates a case where nominal and available
orderings genuinely disagree -- the reason
:class:`~pyaisc360.core.result.StrengthResult` compares on available strength.

Precision: 1e-12 for the algebra (Eqs. D2-1, D2-2, D3-1 are products), and the
published Table D3.1 factors asserted exactly, since they are printed constants
rather than rounded catalog figures.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from pyaisc360 import chapter_d as D
from pyaisc360.core.config import Basis
from pyaisc360.core.enums import LimitState
from pyaisc360.core.exceptions import GeometryError, OutOfScopeError
from pyaisc360.materials import A36
from pyaisc360.sections import SectionAdapter


def plate(Ag: float) -> SectionAdapter:
    return SectionAdapter.from_mapping({"name": f"PL {Ag} in^2", "Ag": Ag})


# ===========================================================================
# Sect. D1
# ===========================================================================
class TestD1Slenderness:
    def test_there_is_no_maximum_slenderness(self) -> None:
        """Sect. D1, p. 16.1-26 opens: "There is no maximum slenderness limit for
        members in tension." The 300 is a User Note suggestion only."""
        member = D.TensionMember(plate(2.5), A36, An=2.5, L_over_r=500.0)
        result = D.tensile_strength(member)
        assert result.available > 0.0
        assert "advisory only" in result.governing.note

    def test_slenderness_below_300_is_not_flagged(self) -> None:
        member = D.TensionMember(plate(2.5), A36, An=2.5, L_over_r=250.0)
        assert "advisory" not in D.tensile_strength(member).governing.note

    def test_the_advisory_limit_value(self) -> None:
        assert D.SLENDERNESS_ADVISORY_LIMIT == 300.0


# ===========================================================================
# Sect. D2
# ===========================================================================
class TestD2Factors:
    def test_yielding_and_rupture_have_different_factors(self) -> None:
        """Sect. D2, p. 16.1-28: yielding 0.90/1.67, rupture 0.75/2.00.

        The only member chapter where the two limit states differ. Yielding of
        the gross section is ductile and redistributes; net-section rupture is
        sudden, so it is insured more heavily.
        """
        assert (D.PHI_T_YIELDING, D.OMEGA_T_YIELDING) == (0.90, 1.67)
        assert (D.PHI_T_RUPTURE, D.OMEGA_T_RUPTURE) == (0.75, 2.00)

    def test_both_pairs_are_calibrated(self) -> None:
        """Omega = 1.5/phi holds for both pairs, as everywhere in the Specification."""
        assert pytest.approx(1.5, abs=0.01) == D.PHI_T_YIELDING * D.OMEGA_T_YIELDING
        assert pytest.approx(1.5, rel=1e-12) == D.PHI_T_RUPTURE * D.OMEGA_T_RUPTURE

    def test_factors_travel_on_the_limit_states(self) -> None:
        result = D.tensile_strength(D.TensionMember(plate(2.5), A36, An=1.5))
        by_state = {s.limit_state: (s.phi, s.omega) for s in result.limit_states}
        assert by_state[LimitState.TENSILE_YIELDING] == (0.90, 1.67)
        assert by_state[LimitState.TENSILE_RUPTURE] == (0.75, 2.00)


class TestD2Strength:
    def test_yielding_is_Fy_Ag(self) -> None:
        """Eq. D2-1."""
        assert D.tensile_yielding(36.0, 2.5) == pytest.approx(90.0, rel=1e-12)

    def test_rupture_is_Fu_Ae(self) -> None:
        """Eq. D2-2."""
        assert D.tensile_rupture(58.0, 1.5) == pytest.approx(87.0, rel=1e-12)

    def test_effective_net_area(self) -> None:
        """Eq. D3-1: Ae = An*U."""
        assert D.effective_net_area(1.5, 0.85) == pytest.approx(1.275, rel=1e-12)

    def test_plate_with_two_bolts(self) -> None:
        """A36 plate 1/2 x 5 with two 7/8-in. bolts.

        Ag  = 0.5 * 5   = 2.5 in^2
        An  = 2.5 - 2*(0.875 + 0.125)*0.5 = 1.5 in^2
        phi*Pn yielding = 0.90 * 36 * 2.5      = 81.00 kips
        phi*Pn rupture  = 0.75 * 58 * 1.5      = 65.25 kips  <- governs
        """
        result = D.tensile_strength(D.TensionMember(plate(2.5), A36, An=1.5))
        assert result.limit_state is LimitState.TENSILE_RUPTURE
        assert result.available == pytest.approx(65.25, rel=1e-12)

    def test_welded_member_yields(self) -> None:
        """With no holes, An = Ag and yielding governs comfortably."""
        result = D.tensile_strength(D.TensionMember(plate(2.5), A36, An=2.5))
        assert result.limit_state is LimitState.TENSILE_YIELDING
        assert result.available == pytest.approx(81.0, rel=1e-12)

    def test_An_defaults_to_Ag(self) -> None:
        assert D.tensile_strength(
            D.TensionMember(plate(2.5), A36)
        ).available == pytest.approx(81.0, rel=1e-12)

    def test_net_area_above_gross_is_rejected(self) -> None:
        with pytest.raises(GeometryError, match="exceeds gross"):
            D.tensile_strength(D.TensionMember(plate(2.5), A36, An=3.0))

    def test_lrfd_and_asd_share_one_nominal_strength(self) -> None:
        member = D.TensionMember(plate(2.5), A36, An=1.5)
        lrfd = D.tensile_strength(member)
        asd = D.tensile_strength(member, basis=Basis.ASD)
        assert lrfd.nominal == pytest.approx(asd.nominal, rel=1e-12)
        assert asd.available == pytest.approx(87.0 / 2.00, rel=1e-12)


class TestOrderingInversion:
    """Nominal and available orderings can disagree -- the reason for per-state factors.

    They invert when ``0.833 < Fy*Ag/(Fu*Ae) < 1.0``: rupture then has the
    **larger** nominal strength but the **smaller** design strength.
    """

    def test_the_inversion_band_endpoints(self) -> None:
        """0.75/0.90 = 0.8333 -- below it yielding wins both ways, above it
        rupture wins both ways, and between them the orderings differ."""
        assert pytest.approx(0.83333, abs=1e-5) == D.PHI_T_RUPTURE / D.PHI_T_YIELDING

    def test_orderings_disagree_inside_the_band(self) -> None:
        """A36, Ag = 2.5, Ae = 1.7: Fy*Ag/(Fu*Ae) = 90/98.6 = 0.913, inside.

        nominal:   yielding 90.0  <  rupture 98.6   -> yielding is lower
        available: yielding 81.0  >  rupture 73.95  -> RUPTURE is lower
        """
        Ag, Ae = 2.5, 1.7
        nominal_yield, nominal_rupture = 36.0 * Ag, 58.0 * Ae
        assert nominal_yield < nominal_rupture
        assert 0.8333 < nominal_yield / nominal_rupture < 1.0

        result = D.tensile_strength(D.TensionMember(plate(Ag), A36, An=Ae))
        assert result.limit_state is LimitState.TENSILE_RUPTURE
        assert result.available == pytest.approx(0.75 * nominal_rupture, rel=1e-12)
        assert result.available < 0.90 * nominal_yield

    def test_governing_by_nominal_would_pick_the_wrong_state(self) -> None:
        """Guard against a regression to min-by-nominal."""
        result = D.tensile_strength(D.TensionMember(plate(2.5), A36, An=1.7))
        lowest_nominal = min(result.limit_states, key=lambda s: s.nominal)
        assert lowest_nominal.limit_state is LimitState.TENSILE_YIELDING
        assert result.governing.limit_state is LimitState.TENSILE_RUPTURE

    def test_the_inversion_also_holds_under_asd(self) -> None:
        result = D.tensile_strength(
            D.TensionMember(plate(2.5), A36, An=1.7), basis=Basis.ASD
        )
        assert result.limit_state is LimitState.TENSILE_RUPTURE
        assert result.available == pytest.approx(58.0 * 1.7 / 2.00, rel=1e-12)

    def test_report_shows_both_columns_when_factors_differ(self) -> None:
        text = D.tensile_strength(D.TensionMember(plate(2.5), A36, An=1.7)).report()
        assert "0.90" in text and "0.75" in text
        assert "governs" in text


# ===========================================================================
# Table D3.1
# ===========================================================================
class TestTableD3_1:
    def test_case_1_is_unity(self) -> None:
        U, note = D.shear_lag_factor(D.ShearLagCase.ALL_ELEMENTS_CONNECTED)
        assert U == 1.0
        assert "case 1" in note

    def test_case_2_general_expression(self) -> None:
        """U = 1 - xbar/l."""
        U, _ = D.shear_lag_factor(
            D.ShearLagCase.SOME_ELEMENTS_CONNECTED, xbar=1.5, length=9.0
        )
        assert pytest.approx(1.0 - 1.5 / 9.0, rel=1e-12) == U

    def test_case_2_rejects_eccentricity_above_the_connection_length(self) -> None:
        with pytest.raises(OutOfScopeError, match="Lengthen the connection"):
            D.shear_lag_factor(D.ShearLagCase.SOME_ELEMENTS_CONNECTED, xbar=9.0, length=9.0)

    def test_case_3_warns_that_An_changes_meaning(self) -> None:
        """Case 3 gives U = 1.0 but redefines An as the DIRECTLY CONNECTED area.

        Taking U = 1.0 and the full net section would badly overstate Ae.
        """
        U, note = D.shear_lag_factor(D.ShearLagCase.TRANSVERSE_WELDS_ONLY)
        assert U == 1.0
        assert "DIRECTLY CONNECTED" in note

    def test_case_4_longitudinal_welds(self) -> None:
        """U = 3l^2/(3l^2 + w^2)*(1 - xbar/l), l = (l1 + l2)/2. New in 2016."""
        U = D.shear_lag_case_4(10.0, 10.0, 6.0, 0.0)
        assert pytest.approx(3 * 100.0 / (3 * 100.0 + 36.0), rel=1e-12) == U

    def test_case_4_handles_unequal_weld_lengths(self) -> None:
        """The 2016 addition: unequal l1 and l2 average, so the pair is weaker
        than two welds of the longer length."""
        equal = D.shear_lag_case_4(12.0, 12.0, 6.0, 0.0)
        unequal = D.shear_lag_case_4(12.0, 8.0, 6.0, 0.0)
        assert unequal < equal
        assert unequal == pytest.approx(D.shear_lag_case_4(10.0, 10.0, 6.0, 0.0), rel=1e-12)

    def test_case_4_improves_with_longer_welds(self) -> None:
        assert D.shear_lag_case_4(20.0, 20.0, 6.0, 0.0) > D.shear_lag_case_4(
            8.0, 8.0, 6.0, 0.0
        )

    def test_case_5_round_hss_full_effectiveness(self) -> None:
        """l >= 1.3*D gives U = 1.0."""
        assert D.shear_lag_case_5(6.0, 7.8) == 1.0
        assert D.shear_lag_case_5(6.0, 12.0) == 1.0

    def test_case_5_uses_D_over_pi_as_the_eccentricity(self) -> None:
        """xbar = D/pi is the centroid of a half-circumference."""
        D_, length = 6.0, 7.0
        assert D.shear_lag_case_5(D_, length) == pytest.approx(
            1.0 - (D_ / math.pi) / length, rel=1e-12
        )

    def test_case_5_below_D_is_out_of_scope(self) -> None:
        with pytest.raises(OutOfScopeError, match="l >= D"):
            D.shear_lag_case_5(6.0, 5.0)

    def test_case_6_two_side_plates_beat_a_single_gusset(self) -> None:
        """The two-plate eccentricity B^2/(4(B+H)) is always below the
        single-gusset (B^2 + 2BH)/(4(B+H))."""
        single = D.shear_lag_case_6(6.0, 8.0, 12.0, two_side_plates=False)
        double = D.shear_lag_case_6(6.0, 8.0, 12.0, two_side_plates=True)
        assert double > single

    def test_case_6_B_and_H_are_not_interchangeable(self) -> None:
        """B is measured 90 degrees to the connection plane, H in it."""
        as_given = D.shear_lag_case_6(6.0, 10.0, 14.0)
        swapped = D.shear_lag_case_6(10.0, 6.0, 14.0)
        assert as_given != pytest.approx(swapped, rel=1e-6)

    def test_case_6_below_H_is_out_of_scope(self) -> None:
        with pytest.raises(OutOfScopeError, match="l >= H"):
            D.shear_lag_case_6(6.0, 10.0, 9.0)

    @pytest.mark.parametrize(
        ("bf", "d", "expected"),
        [(10.0, 14.0, 0.90), (8.0, 14.0, 0.85)],
    )
    def test_case_7_flange_connected(self, bf: float, d: float, expected: float) -> None:
        """U = 0.90 when bf >= 2/3 d, else 0.85. Published constants."""
        U, _ = D.shear_lag_factor(
            D.ShearLagCase.W_SHAPE_BOLTED, bf=bf, d=d, fasteners_per_line=3
        )
        assert expected == U

    def test_case_7_boundary_is_exactly_two_thirds(self) -> None:
        d = 15.0
        U, _ = D.shear_lag_factor(
            D.ShearLagCase.W_SHAPE_BOLTED, bf=2.0 / 3.0 * d, d=d, fasteners_per_line=3
        )
        assert U == 0.90

    def test_case_7_web_connected(self) -> None:
        U, _ = D.shear_lag_factor(
            D.ShearLagCase.W_SHAPE_BOLTED, web_connected=True, fasteners_per_line=4
        )
        assert U == 0.70

    def test_case_7_requires_enough_fasteners(self) -> None:
        with pytest.raises(OutOfScopeError, match="three or more"):
            D.shear_lag_factor(
                D.ShearLagCase.W_SHAPE_BOLTED, bf=10.0, d=14.0, fasteners_per_line=2
            )
        with pytest.raises(OutOfScopeError, match="four or more"):
            D.shear_lag_factor(
                D.ShearLagCase.W_SHAPE_BOLTED, web_connected=True, fasteners_per_line=3
            )

    @pytest.mark.parametrize(("fasteners", "expected"), [(4, 0.80), (5, 0.80), (3, 0.60)])
    def test_case_8_angles(self, fasteners: int, expected: float) -> None:
        U, _ = D.shear_lag_factor(D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=fasteners)
        assert expected == U

    def test_case_8_with_two_fasteners_directs_to_case_2(self) -> None:
        with pytest.raises(OutOfScopeError, match="use case 2"):
            D.shear_lag_factor(D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=2)

    def test_case_2_supersedes_case_8_when_larger(self) -> None:
        """Table D3.1: "If U is calculated per Case 2, the larger value is
        permitted to be used"."""
        U, note = D.shear_lag_factor(
            D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=3, xbar=0.9, length=9.0
        )
        assert pytest.approx(0.90, rel=1e-12) == U
        assert "superseded by case 2" in note

    def test_case_2_does_not_supersede_when_smaller(self) -> None:
        U, note = D.shear_lag_factor(
            D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=4, xbar=3.0, length=9.0
        )
        assert U == 0.80
        assert "superseded" not in note


class TestSectD3Floor:
    def test_open_sections_get_the_connected_area_floor(self) -> None:
        """Sect. D3, p. 16.1-28: for W, M, S, C, HP, WT, ST and angles, U "need
        not be less than the ratio of the gross area of the connected element(s)
        to the member gross area"."""
        U, note = D.shear_lag_factor(
            D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=3, connected_area_ratio=0.72
        )
        assert pytest.approx(0.72, rel=1e-12) == U
        assert "Sect. D3 floor" in note

    def test_the_floor_never_pushes_U_above_one(self) -> None:
        U, _ = D.shear_lag_factor(
            D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=3, connected_area_ratio=1.4
        )
        assert U == 1.0

    def test_the_floor_is_opt_in_because_it_excludes_HSS_and_plates(self) -> None:
        """Sect. D3: "This provision does not apply to closed sections, such as
        HSS sections, nor to plates." So it is an explicit argument, never inferred.
        """
        without, _ = D.shear_lag_factor(D.ShearLagCase.ANGLE_BOLTED, fasteners_per_line=3)
        assert without == 0.60


# ===========================================================================
# Sect. D5 -- pin-connected members
# ===========================================================================
class TestD5PinConnected:
    def test_effective_edge_distance(self) -> None:
        """be = 2t + 0.63, capped at the actual edge distance."""
        assert D.pin_effective_edge_distance(0.5, 3.0) == pytest.approx(1.63, rel=1e-12)
        assert D.pin_effective_edge_distance(0.5, 1.2) == 1.2

    def test_tensile_rupture_counts_both_sides(self) -> None:
        """Eq. D5-1: Pn = Fu*(2*t*be) -- the two sides of the pin hole."""
        assert D.pin_tensile_rupture(58.0, 0.5, 1.63) == pytest.approx(
            58.0 * 2 * 0.5 * 1.63, rel=1e-12
        )

    def test_shear_rupture_area(self) -> None:
        """Eq. D5-2: Pn = 0.6*Fu*Asf, Asf = 2*t*(a + d/2)."""
        got = D.pin_shear_rupture(58.0, 0.5, 2.0, 1.0)
        assert got == pytest.approx(0.6 * 58.0 * 2 * 0.5 * (2.0 + 0.5), rel=1e-12)

    def test_dimensional_requirements_pass(self) -> None:
        assert D.check_pin_dimensional_requirements(6.0, 1.63, 1.0, 3.0) == []

    def test_narrow_plate_is_rejected(self) -> None:
        with pytest.raises(OutOfScopeError, match="plate width"):
            D.check_pin_dimensional_requirements(3.0, 1.63, 1.0, 3.0)

    def test_short_extension_is_rejected(self) -> None:
        with pytest.raises(OutOfScopeError, match="extension"):
            D.check_pin_dimensional_requirements(6.0, 1.63, 1.0, 1.0)

    def test_violations_can_be_collected_without_raising(self) -> None:
        violations = D.check_pin_dimensional_requirements(3.0, 1.63, 1.0, 1.0, strict=False)
        assert len(violations) == 2


# ===========================================================================
# Published benchmarks
# ===========================================================================
_EXAMPLES = Path(__file__).parent / "design_examples" / "chapter_d.json"


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(_EXAMPLES.read_text(encoding="utf-8"))
    return [case for case in data.get("cases", []) if not case.get("skip")]


class TestPublishedBenchmarks:
    @pytest.mark.design_example
    @pytest.mark.parametrize("case", _load_cases(), ids=lambda c: str(c.get("id", "?")))
    def test_matches_the_published_values(self, case: dict[str, Any]) -> None:
        from pyaisc360.materials import grade

        member = D.TensionMember(
            plate(float(case["Ag"])), grade(str(case["grade"])),
            An=float(case["An"]), U=float(case["U"]),
        )
        result = D.tensile_strength(member)
        assert result.available == pytest.approx(
            float(case["phiPn_kips"]), rel=float(case.get("tol", 0.005))
        )
        assert result.limit_state.value == case["governing"]

    def test_every_case_declares_its_source(self) -> None:
        for case in _load_cases():
            assert case.get("source"), f"case {case.get('id')!r} has no source"

    def test_the_inversion_case_is_present(self) -> None:
        """The ordering-inversion case must stay in the suite -- it is the only
        one that would catch a regression to min-by-nominal."""
        ids = " ".join(str(c["id"]) for c in _load_cases())
        assert "inversion" in ids


class TestTableD3_1PublishedConstants:
    """Every tabulated U in Table D3.1, asserted exactly.

    These are printed constants in the Specification, not rounded catalog
    figures, so they are held to equality rather than a tolerance.
    """

    @pytest.mark.design_example
    @pytest.mark.parametrize(
        ("case", "kwargs", "expected"),
        [
            (D.ShearLagCase.ALL_ELEMENTS_CONNECTED, {}, 1.00),
            (D.ShearLagCase.TRANSVERSE_WELDS_ONLY, {}, 1.00),
            (D.ShearLagCase.W_SHAPE_BOLTED,
             dict(bf=12.0, d=14.0, fasteners_per_line=3), 0.90),
            (D.ShearLagCase.W_SHAPE_BOLTED,
             dict(bf=7.0, d=14.0, fasteners_per_line=3), 0.85),
            (D.ShearLagCase.W_SHAPE_BOLTED,
             dict(web_connected=True, fasteners_per_line=4), 0.70),
            (D.ShearLagCase.ANGLE_BOLTED, dict(fasteners_per_line=4), 0.80),
            (D.ShearLagCase.ANGLE_BOLTED, dict(fasteners_per_line=3), 0.60),
        ],
    )
    def test_published_factor(
        self, case: D.ShearLagCase, kwargs: dict[str, Any], expected: float
    ) -> None:
        """Source: Table D3.1, pp. 16.1-30 to 16.1-31."""
        U, _ = D.shear_lag_factor(case, **kwargs)
        assert expected == U
