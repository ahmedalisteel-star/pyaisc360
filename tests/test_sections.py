"""The section-property contract: protocols, adapter, unit conversion, derivations."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pyaisc360.core.exceptions import GeometryError, MissingPropertyError
from pyaisc360.sections import (
    IShapedSection,
    SectionAdapter,
    available_properties,
    derive_c_channel,
    derive_ho,
    derive_rts,
    require_properties,
)


@dataclass
class NativeShape:
    """An object that already speaks AISC -- should satisfy the protocol as-is."""

    name: str
    Ag: float
    d: float
    bf: float
    tf: float
    tw: float
    Ix: float
    Iy: float
    Sx: float
    Sy: float
    Zx: float
    Zy: float
    rx: float
    ry: float
    J: float
    Cw: float
    rts: float
    ho: float


class ForeignShape:
    """A third-party record using its own names -- needs the adapter."""

    def __init__(self) -> None:
        self.designation = "W18X50"
        self.gross_area = 14.7
        self.depth = 18.0
        self.flange_width = 7.50
        self.flange_thickness = 0.570
        self.Wpl_y = 101.0
        self.I_y = 40.1
        self.warping_constant = 3040.0
        self.S_x = 88.9


class TestProtocolConformance:
    def test_native_object_satisfies_the_protocol(
        self, w18x50_props: dict[str, float]
    ) -> None:
        fields = NativeShape.__annotations__
        shape = NativeShape(
            name="W18X50", **{k: v for k, v in w18x50_props.items() if k in fields}
        )
        assert isinstance(shape, IShapedSection)

    def test_incomplete_object_does_not(self) -> None:
        @dataclass
        class OnlyArea:
            name: str
            Ag: float

        assert not isinstance(OnlyArea("X", 10.0), IShapedSection)


class TestAdapterResolution:
    def test_reads_canonical_names(self, w18x50: SectionAdapter) -> None:
        assert w18x50.Ag == 14.7
        assert w18x50.Zx == 101.0

    def test_resolves_aliases(self) -> None:
        sec = SectionAdapter(ForeignShape())
        assert sec.name == "W18X50"
        assert sec.Ag == 14.7
        assert sec.d == 18.0
        assert sec.Zx == 101.0
        assert sec.Cw == 3040.0

    def test_explicit_overrides_win(self) -> None:
        sec = SectionAdapter(
            {"gross_area_in2": 14.7, "A": 999.0}, overrides={"Ag": "gross_area_in2"}
        )
        assert sec.Ag == 14.7

    def test_bare_A_is_accepted_for_gross_area(self) -> None:
        assert SectionAdapter.from_mapping({"A": 14.7}).Ag == 14.7

    def test_missing_property_raises_with_guidance(self) -> None:
        sec = SectionAdapter.from_mapping({"Ag": 14.7})
        with pytest.raises(MissingPropertyError, match="overrides"):
            _ = sec.J

    def test_has_reports_availability(self, w18x50: SectionAdapter) -> None:
        assert w18x50.has("Ag", "Zx", "rts")
        assert not w18x50.has("Ag", "eo")

    def test_values_are_cached(self, w18x50: SectionAdapter) -> None:
        assert w18x50.Ag is w18x50.Ag

    def test_to_dict(self, w18x50: SectionAdapter) -> None:
        assert w18x50.to_dict("Ag", "Zx") == {"Ag": 14.7, "Zx": 101.0}


class TestMetricConversion:
    def test_lengths_convert(self, w18x50_metric: SectionAdapter) -> None:
        assert w18x50_metric.d == pytest.approx(18.0)

    def test_areas_convert_by_the_square(self, w18x50_metric: SectionAdapter) -> None:
        assert w18x50_metric.Ag == pytest.approx(14.7)

    def test_second_moments_convert_by_the_fourth_power(
        self, w18x50_metric: SectionAdapter
    ) -> None:
        assert w18x50_metric.Iy == pytest.approx(40.1)

    def test_section_moduli_convert_by_the_cube(self, w18x50_metric: SectionAdapter) -> None:
        assert w18x50_metric.Sx == pytest.approx(88.9)

    def test_warping_constant_converts_by_the_sixth_power(
        self, w18x50_metric: SectionAdapter
    ) -> None:
        assert w18x50_metric.Cw == pytest.approx(3040.0)

    def test_derivation_happens_before_conversion(self, w18x50_metric: SectionAdapter) -> None:
        """rts derived from raw metric Iy/Cw/Sx must still land in inches."""
        assert w18x50_metric.rts == pytest.approx(1.98, abs=0.01)

    def test_ratios_are_not_converted(self) -> None:
        sec = SectionAdapter.from_mapping({"bf": 190.5, "tf": 14.478}, metric=True)
        assert sec.bf_2tf == pytest.approx(6.58, abs=0.01)

    def test_unknown_dimension_refuses_to_guess(self) -> None:
        sec = SectionAdapter.from_mapping({"weird_prop": 5.0}, metric=True)
        with pytest.raises(MissingPropertyError, match="dimension is unknown"):
            _ = sec.weird_prop


class TestDerivations:
    def test_ho_is_d_minus_tf(self) -> None:
        """W18x50: d = 18.0, tf = 0.570 -> ho = 17.43 vs 17.4 tabulated."""
        assert derive_ho(18.0, 0.570) == pytest.approx(17.43)

    def test_ho_derived_when_absent(self) -> None:
        sec = SectionAdapter.from_mapping({"d": 18.0, "tf": 0.570})
        assert sec.ho == pytest.approx(17.43)

    def test_tabulated_ho_wins_over_derivation(self, w18x50: SectionAdapter) -> None:
        assert w18x50.ho == 17.4

    def test_rts_matches_the_published_value(self) -> None:
        """Eq. F2-7 for W18x50: sqrt(sqrt(40.1*3040)/88.9) = 1.98 in."""
        assert derive_rts(40.1, 3040.0, 88.9) == pytest.approx(1.98, abs=0.005)

    def test_rts_matches_for_w14x90(self) -> None:
        assert derive_rts(362.0, 16000.0, 143.0) == pytest.approx(4.10, abs=0.01)

    def test_rts_user_note_identity(self) -> None:
        """User Note to Eq. F2-7, p. 16.1-48: for a doubly symmetric I-shape with
        rectangular flanges, Cw = Iy*ho^2/4, so rts^2 = Iy*ho/(2*Sx)."""
        Iy, ho, Sx = 40.1, 17.43, 88.9
        Cw = Iy * ho**2 / 4.0
        assert derive_rts(Iy, Cw, Sx) ** 2 == pytest.approx(Iy * ho / (2.0 * Sx), rel=1e-12)

    def test_c_is_unity_for_a_doubly_symmetric_shape(self) -> None:
        """Eq. F2-8a gives c = 1; Eq. F2-8b must reduce to it when Cw = Iy*ho^2/4."""
        Iy, ho = 40.1, 17.43
        assert derive_c_channel(ho, Iy, Iy * ho**2 / 4.0) == pytest.approx(1.0, rel=1e-12)

    @pytest.mark.parametrize(("d", "tf"), [(0.0, 0.5), (18.0, 0.0), (18.0, 9.0), (-1.0, 0.5)])
    def test_impossible_geometry_raises(self, d: float, tf: float) -> None:
        with pytest.raises(GeometryError):
            derive_ho(d, tf)

    def test_rts_rejects_non_positive_inputs(self) -> None:
        with pytest.raises(GeometryError, match="must be positive"):
            derive_rts(0.0, 3040.0, 88.9)


class TestPropertyGuards:
    def test_require_lists_everything_missing_at_once(self) -> None:
        sec = SectionAdapter.from_mapping({"Ag": 14.7, "name": "partial"})
        with pytest.raises(MissingPropertyError) as exc:
            require_properties(sec, "Ag", "J", "Cw", "Zx")
        message = str(exc.value)
        assert "J" in message and "Cw" in message and "Zx" in message

    def test_require_returns_values_when_complete(self, w18x50: SectionAdapter) -> None:
        assert require_properties(w18x50, "Ag", "Zx") == {"Ag": 14.7, "Zx": 101.0}

    def test_available_skips_missing_without_raising(self) -> None:
        sec = SectionAdapter.from_mapping({"Ag": 14.7})
        assert available_properties(sec, "Ag", "J", "Cw") == {"Ag": 14.7}
