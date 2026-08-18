"""Steel grades, elastic constants and the Sect. B4.2 design wall thickness."""

from __future__ import annotations

import pytest

from pyaisc360 import materials
from pyaisc360.core.enums import ProductForm
from pyaisc360.core.exceptions import MaterialError


class TestElasticConstants:
    def test_modulus_of_elasticity(self) -> None:
        """AISC 360-16, Sect. E3, p. 16.1-35: E = 29,000 ksi."""
        assert materials.E_STEEL == 29000.0

    def test_shear_modulus(self) -> None:
        """AISC 360-16, Sect. E4, p. 16.1-36: G = 11,200 ksi."""
        assert materials.G_STEEL == 11200.0

    def test_moduli_are_consistent_with_poisson(self) -> None:
        """G = E / (2*(1+nu)) to within the rounding of the published values."""
        implied = materials.E_STEEL / (2.0 * (1.0 + materials.POISSON))
        assert pytest.approx(implied, rel=0.005) == materials.G_STEEL


class TestGrades:
    def test_a992_is_the_w_shape_default(self) -> None:
        assert (materials.A992.Fy, materials.A992.Fu) == (50.0, 65.0)

    def test_a36(self) -> None:
        assert (materials.A36.Fy, materials.A36.Fu) == (36.0, 58.0)

    @pytest.mark.parametrize(
        ("spelling", "expected"),
        [
            ("A992", "A992"),
            ("a992", "A992"),
            ("A572 Gr. 50", "A572 Gr. 50"),
            ("a572-50", "A572 Gr. 50"),
            ("A572GR50", "A572 Gr. 50"),
            ("a36", "A36"),
            ("A1085", "A1085"),
        ],
    )
    def test_lookup_is_tolerant(self, spelling: str, expected: str) -> None:
        assert materials.grade(spelling).name == expected

    def test_unknown_grade_raises(self) -> None:
        with pytest.raises(MaterialError, match="unknown steel grade"):
            materials.grade("A999")

    def test_every_grade_has_fu_at_or_above_fy(self) -> None:
        for steel in materials.GRADES.values():
            assert steel.Fu >= steel.Fy, steel.name

    def test_transposed_strengths_are_rejected(self) -> None:
        with pytest.raises(MaterialError, match="transposed"):
            materials.Steel("bogus", Fy=65.0, Fu=50.0, standard="none")

    def test_a500_round_and_rectangular_differ(self) -> None:
        """A500 Gr. B: Fy = 46 ksi rectangular but 42 ksi round -- a classic trap."""
        assert materials.A500_B_RECT.Fy == 46.0
        assert materials.A500_B_ROUND.Fy == 42.0
        assert materials.A500_C_RECT.Fy == 50.0
        assert materials.A500_C_ROUND.Fy == 46.0


class TestProductForms:
    def test_a992_is_shapes_only(self) -> None:
        """Sect. A3.1a, p. 16.1-7 lists A992 under hot-rolled shapes only."""
        materials.A992.require_form(ProductForm.SHAPE)
        with pytest.raises(MaterialError, match="not approved"):
            materials.A992.require_form(ProductForm.HSS)

    def test_a36_covers_shapes_plates_and_bars(self) -> None:
        for form in (ProductForm.SHAPE, ProductForm.PLATE, ProductForm.BAR):
            materials.A36.require_form(form)

    def test_grades_for_hss_are_all_hss_standards(self) -> None:
        hss = materials.grades_for(ProductForm.HSS)
        assert materials.A500_C_RECT in hss
        assert materials.A992 not in hss


class TestDesignWallThickness:
    def test_a500_takes_the_093_factor(self) -> None:
        """Sect. B4.2, p. 16.1-20: 0.93 x nominal for HSS other than A1065/A1085."""
        assert materials.design_wall_thickness(0.500, materials.A500_C_RECT) == pytest.approx(0.465)

    def test_a1085_takes_the_full_thickness(self) -> None:
        assert materials.design_wall_thickness(0.500, materials.A1085) == 0.500

    def test_a1065_takes_the_full_thickness(self) -> None:
        assert materials.design_wall_thickness(0.500, materials.A1065_50) == 0.500

    def test_box_section_takes_the_full_thickness_regardless_of_grade(self) -> None:
        got = materials.design_wall_thickness(0.500, materials.A36, box_section=True)
        assert got == 0.500

    def test_non_hss_grade_raises(self) -> None:
        with pytest.raises(MaterialError, match="not approved"):
            materials.design_wall_thickness(0.500, materials.A992)

    def test_non_positive_thickness_raises(self) -> None:
        with pytest.raises(MaterialError, match="must be positive"):
            materials.design_wall_thickness(0.0, materials.A500_C_RECT)

    def test_the_factor_matters(self) -> None:
        """Ignoring it overstates a wall area by 7.5% -- worth a dedicated test."""
        nominal, design = 0.500, materials.design_wall_thickness(0.500, materials.A500_B_RECT)
        assert (nominal - design) / nominal == pytest.approx(0.07, abs=1e-9)


class TestSelfWeight:
    def test_w18x50_weighs_50_plf(self) -> None:
        """Ag = 14.7 in.^2 at 490 pcf -> 50.0 lb/ft, which is the shape's name."""
        assert materials.A992.weight_per_foot(14.7) == pytest.approx(50.0, abs=0.05)

    def test_w14x90_weighs_90_plf(self) -> None:
        assert materials.A992.weight_per_foot(26.5) == pytest.approx(90.2, abs=0.3)
