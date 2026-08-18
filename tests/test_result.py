"""Result objects and the traceability index."""

from __future__ import annotations

import pytest

from pyaisc360.core.citations import (
    UnknownEquationError,
    cite,
    known_equations,
    spec_index,
)
from pyaisc360.core.config import Basis, DesignSettings
from pyaisc360.core.enums import LimitState
from pyaisc360.core.exceptions import AISC360Error
from pyaisc360.core.result import LimitStateResult, StrengthResult


@pytest.fixture
def yielding() -> LimitStateResult:
    return LimitStateResult(
        limit_state=LimitState.PLASTIC_MOMENT,
        nominal=5050.0,
        citation=cite("F2-1"),
        detail={"Zx": 101.0, "Fy": 50.0},
    )


@pytest.fixture
def ltb() -> LimitStateResult:
    return LimitStateResult(
        limit_state=LimitState.LATERAL_TORSIONAL_BUCKLING,
        nominal=4550.0,
        citation=cite("F2-2"),
        note="Lp < Lb <= Lr, inelastic LTB",
    )


@pytest.fixture
def result(yielding: LimitStateResult, ltb: LimitStateResult) -> StrengthResult:
    return StrengthResult.build(
        "Mn", "kip-in.", [yielding, ltb], phi=0.90, omega=1.67, basis=Basis.LRFD
    )


class TestCitations:
    def test_resolves_a_known_equation(self) -> None:
        c = cite("F2-2")
        assert (c.equation, c.section, c.page) == ("F2-2", "F2", 47)

    def test_text_is_report_ready(self) -> None:
        assert cite("E3-4").text == "AISC 360-16, Eq. E3-4 (Sect. E3), p. 16.1-36"

    def test_appendix_equations_resolve(self) -> None:
        assert cite("A-8-1").section == "A8"

    def test_unknown_label_raises(self) -> None:
        with pytest.raises(UnknownEquationError, match="not a numbered equation"):
            cite("F2-99")

    def test_index_covers_the_whole_specification(self) -> None:
        """373 = 372 scanned + 1 recovered from a known extraction defect.

        The dump prints "(K2-2a)" twice in Table K2.1 where the second label is
        "(K2-2b)"; ``tools/scan_spec.py::EXTRACTION_CORRECTIONS`` restores it
        with the evidence recorded.
        """
        labels = known_equations()
        assert len(labels) == 373
        for expected in ("B3-1", "B3-2", "D2-1", "E3-2", "F1-1", "F2-8b", "G2-1", "H1-1a"):
            assert expected in labels

    def test_extraction_corrections_are_recorded_and_applied(self) -> None:
        """A correction is only legitimate if it says what it changed and why."""
        corrections = spec_index()["corrections"]
        assert corrections
        for entry in corrections:
            assert entry["dump_says"] != entry["actually"]
            assert len(str(entry["why"])) > 80, "a correction must justify itself"
            assert entry["actually"] in known_equations()


class TestLimitStateResult:
    def test_rejects_negative_strength(self) -> None:
        with pytest.raises(AISC360Error, match="negative nominal strength"):
            LimitStateResult(LimitState.YIELDING, -1.0, cite("F2-1"))

    def test_zero_strength_is_allowed(self) -> None:
        """A fully slender element can legitimately contribute nothing."""
        assert LimitStateResult(LimitState.LOCAL_BUCKLING, 0.0, cite("E7-1")).nominal == 0.0


class TestStrengthResult:
    def test_governing_is_the_lowest(self, result: StrengthResult) -> None:
        assert result.limit_state is LimitState.LATERAL_TORSIONAL_BUCKLING
        assert result.nominal == 4550.0

    def test_ties_resolve_to_the_first_listed(self) -> None:
        """At Lb = Lp both give Mp; the Specification lists yielding first."""
        a = LimitStateResult(LimitState.PLASTIC_MOMENT, 5050.0, cite("F2-1"))
        b = LimitStateResult(LimitState.LATERAL_TORSIONAL_BUCKLING, 5050.0, cite("F2-2"))
        res = StrengthResult.build("Mn", "kip-in.", [a, b], phi=0.9, omega=1.67, basis=Basis.LRFD)
        assert res.limit_state is LimitState.PLASTIC_MOMENT

    def test_lrfd_available_strength(self, result: StrengthResult) -> None:
        assert result.available == pytest.approx(0.90 * 4550.0)

    def test_asd_available_strength(self, result: StrengthResult) -> None:
        asd = result.as_basis(Basis.ASD)
        assert asd.available == pytest.approx(4550.0 / 1.67)

    def test_switching_basis_keeps_the_nominal_strength(self, result: StrengthResult) -> None:
        assert result.as_basis(Basis.ASD).nominal == result.nominal

    def test_citation_follows_the_governing_state(self, result: StrengthResult) -> None:
        assert result.citation.equation == "F2-2"

    def test_utilization(self, result: StrengthResult) -> None:
        assert result.utilization(2047.5) == pytest.approx(0.5)
        assert result.is_adequate(2047.5)
        assert not result.is_adequate(9999.0)

    def test_float_conversion_gives_available_strength(self, result: StrengthResult) -> None:
        assert float(result) == pytest.approx(4095.0)

    def test_requires_at_least_one_limit_state(self) -> None:
        with pytest.raises(AISC360Error, match="no limit states"):
            StrengthResult.build("Mn", "kip-in.", [], phi=0.9, omega=1.67, basis=Basis.LRFD)

    def test_build_drops_inapplicable_states(self, yielding: LimitStateResult) -> None:
        res = StrengthResult.build(
            "Mn", "kip-in.", [yielding, None], phi=0.9, omega=1.67, basis=Basis.LRFD
        )
        assert len(res.limit_states) == 1

    def test_report_shows_every_state_and_marks_the_governor(
        self, result: StrengthResult
    ) -> None:
        text = result.report(required=3000.0)
        assert "lateral-torsional buckling" in text
        assert "yielding (plastic moment)" in text
        assert "governs" in text
        assert "Eq. F2-2" in text
        assert "OK" in text

    def test_report_flags_an_overstressed_member(self, result: StrengthResult) -> None:
        assert "NOT OK" in result.report(required=99999.0)

    def test_report_shows_omega_under_asd(self, result: StrengthResult) -> None:
        text = result.as_basis(Basis.ASD).report()
        assert "Omega = 1.67" in text
        assert "Mn/Omega" in text


class TestDesignSettings:
    def test_defaults_to_lrfd_and_strict(self) -> None:
        settings = DesignSettings()
        assert settings.basis is Basis.LRFD
        assert settings.strict

    def test_with_basis_returns_a_copy(self) -> None:
        settings = DesignSettings()
        assert settings.with_basis(Basis.ASD).basis is Basis.ASD
        assert settings.basis is Basis.LRFD

    def test_demand_symbols_match_the_specification(self) -> None:
        assert Basis.LRFD.demand_symbol == "Ru"
        assert Basis.ASD.demand_symbol == "Ra"
