"""Structural steel grades and elastic constants.

Two different sources feed this module, and the distinction matters:

* **E, G and Poisson's ratio come from the Specification itself** -- E = 29,000 ksi
  (Sect. E3, p. 16.1-35) and G = 11,200 ksi (Sect. E4, p. 16.1-36).
* **Fy and Fu do not.** Section A3.1a (p. 16.1-7) only *lists which ASTM
  standards are approved*; the strength values belong to those ASTM standards
  (and are tabulated in AISC *Manual* Tables 2-4 and 2-5). They are reproduced
  here as data, and every grade records the standard it came from so a value can
  be checked against its actual source rather than against 360-16.

Grades also record which product forms they are approved for, so applying a
plate grade to an HSS raises instead of quietly producing a number.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .core.enums import ProductForm
from .core.exceptions import MaterialError
from .core.units import Inch, Ksi

__all__ = [
    "Steel",
    "E_STEEL",
    "G_STEEL",
    "POISSON",
    "DENSITY_PCF",
    "GRADES",
    "grade",
    "grades_for",
    "design_wall_thickness",
    # the handful used constantly, exported by name for convenience
    "A36",
    "A992",
    "A572_50",
    "A913_65",
    "A500_B_RECT",
    "A500_B_ROUND",
    "A500_C_RECT",
    "A500_C_ROUND",
    "A1085",
    "A53_B",
]

#: Modulus of elasticity of steel. AISC 360-16, Sect. E3, p. 16.1-35.
E_STEEL: Ksi = 29000.0
#: Shear modulus of elasticity of steel. AISC 360-16, Sect. E4, p. 16.1-36.
G_STEEL: Ksi = 11200.0
#: Poisson's ratio in the elastic range (implied by E and G above).
POISSON: float = 0.3
#: Density of structural steel, lb/ft^3. AISC *Manual* Part 17.
DENSITY_PCF: float = 490.0
#: Coefficient of thermal expansion, per degree F, 70-100 F range.
THERMAL_EXPANSION: float = 6.5e-6

#: HSS standards whose design wall thickness equals the nominal thickness.
#: AISC 360-16, Sect. B4.2, p. 16.1-20.
_FULL_THICKNESS_HSS = frozenset({"ASTM A1065/A1065M", "ASTM A1085/A1085M"})


@dataclass(frozen=True, slots=True)
class Steel:
    """A structural steel grade.

    Attributes
    ----------
    name:
        Short designation used to look the grade up, e.g. ``"A992"``.
    Fy:
        Specified minimum yield stress, ksi.
    Fu:
        Specified minimum tensile strength, ksi.
    standard:
        The ASTM standard Fy and Fu come from -- these values are *not* given in
        AISC 360-16, which only lists the approved standards in Sect. A3.1a.
    forms:
        Product forms this grade is approved for under Sect. A3.1a, p. 16.1-7.
    E, G:
        Elastic and shear moduli, ksi. Per the Specification, not per grade.
    note:
        Anything that qualifies the values, e.g. a thickness range.
    """

    name: str
    Fy: Ksi
    Fu: Ksi
    standard: str
    forms: frozenset[ProductForm] = field(default_factory=lambda: frozenset({ProductForm.SHAPE}))
    E: Ksi = E_STEEL
    G: Ksi = G_STEEL
    poisson: float = POISSON
    density_pcf: float = DENSITY_PCF
    thermal_expansion: float = THERMAL_EXPANSION
    note: str = ""

    def __post_init__(self) -> None:
        if self.Fy <= 0.0:
            raise MaterialError(f"{self.name}: Fy must be positive, got {self.Fy}")
        if self.Fu < self.Fy:
            raise MaterialError(
                f"{self.name}: Fu ({self.Fu} ksi) is below Fy ({self.Fy} ksi); "
                "the values look transposed"
            )
        if self.E <= 0.0 or self.G <= 0.0:
            raise MaterialError(f"{self.name}: E and G must be positive")

    # -- derived ----------------------------------------------------------
    @property
    def density_pci(self) -> float:
        """Density in lb/in.^3 -- the unit that pairs with in.^2 areas."""
        return self.density_pcf / 1728.0

    def weight_per_foot(self, area: float) -> float:
        """Self-weight in lb/ft for a gross area in in.^2."""
        return self.density_pcf * area / 144.0

    def require_form(self, form: ProductForm) -> None:
        """Raise unless this grade is approved for ``form`` (Sect. A3.1a).

        Raises
        ------
        MaterialError
            If the grade is not listed for that product form.
        """
        if form not in self.forms:
            approved = ", ".join(sorted(f.value for f in self.forms))
            raise MaterialError(
                f"{self.name} is not approved for {form.value} under AISC 360-16 "
                f"Sect. A3.1a (approved: {approved})"
            )

    def with_yield(self, Fy: Ksi, Fu: Ksi | None = None, *, note: str = "") -> Steel:
        """A variant at a different specified strength -- e.g. a thickness group."""
        return replace(self, Fy=Fy, Fu=Fu if Fu is not None else self.Fu, note=note)

    def __str__(self) -> str:
        return f"{self.name} (Fy = {self.Fy:g} ksi, Fu = {self.Fu:g} ksi)"


_SHAPE = frozenset({ProductForm.SHAPE})
_SHAPE_PLATE_BAR = frozenset({ProductForm.SHAPE, ProductForm.PLATE, ProductForm.BAR})
_PLATE = frozenset({ProductForm.PLATE})
_HSS = frozenset({ProductForm.HSS})


# ---------------------------------------------------------------------------
# Hot-rolled shapes, plates and bars -- Sect. A3.1a(a), (c), (d), p. 16.1-7
# ---------------------------------------------------------------------------
A36 = Steel("A36", 36.0, 58.0, "ASTM A36/A36M", _SHAPE_PLATE_BAR,
            note="Fu 58-80 ksi; the specified minimum governs design")
A529_50 = Steel("A529 Gr. 50", 50.0, 70.0, "ASTM A529/A529M", _SHAPE_PLATE_BAR)
A529_55 = Steel("A529 Gr. 55", 55.0, 70.0, "ASTM A529/A529M", _SHAPE_PLATE_BAR)
A572_42 = Steel("A572 Gr. 42", 42.0, 60.0, "ASTM A572/A572M", _SHAPE_PLATE_BAR)
A572_50 = Steel("A572 Gr. 50", 50.0, 65.0, "ASTM A572/A572M", _SHAPE_PLATE_BAR)
A572_55 = Steel("A572 Gr. 55", 55.0, 70.0, "ASTM A572/A572M", _SHAPE_PLATE_BAR)
A572_60 = Steel("A572 Gr. 60", 60.0, 75.0, "ASTM A572/A572M", _SHAPE_PLATE_BAR)
A572_65 = Steel("A572 Gr. 65", 65.0, 80.0, "ASTM A572/A572M", _SHAPE_PLATE_BAR)
A588 = Steel("A588", 50.0, 70.0, "ASTM A588/A588M", _SHAPE_PLATE_BAR,
             note="weathering steel; Fy reduces for plate thickness over 4 in.")
A709_36 = Steel("A709 Gr. 36", 36.0, 58.0, "ASTM A709/A709M", _SHAPE_PLATE_BAR)
A709_50 = Steel("A709 Gr. 50", 50.0, 65.0, "ASTM A709/A709M", _SHAPE_PLATE_BAR)
A913_50 = Steel("A913 Gr. 50", 50.0, 65.0, "ASTM A913/A913M", _SHAPE)
A913_60 = Steel("A913 Gr. 60", 60.0, 75.0, "ASTM A913/A913M", _SHAPE)
A913_65 = Steel("A913 Gr. 65", 65.0, 80.0, "ASTM A913/A913M", _SHAPE)
A913_70 = Steel("A913 Gr. 70", 70.0, 90.0, "ASTM A913/A913M", _SHAPE)
A992 = Steel("A992", 50.0, 65.0, "ASTM A992/A992M", _SHAPE,
             note="default for W-shapes; Fy/Fu ratio limited to 0.85")
A1043_36 = Steel("A1043 Gr. 36", 36.0, 58.0, "ASTM A1043/A1043M", _SHAPE_PLATE_BAR)
A1043_50 = Steel("A1043 Gr. 50", 50.0, 65.0, "ASTM A1043/A1043M", _SHAPE_PLATE_BAR)
A242 = Steel("A242", 50.0, 70.0, "ASTM A242/A242M", _PLATE,
             note="Fy 42-50 ksi by thickness group; 50 ksi shown for t <= 3/4 in.")
A283_C = Steel("A283 Gr. C", 30.0, 55.0, "ASTM A283/A283M", _PLATE)
A514 = Steel("A514", 100.0, 110.0, "ASTM A514/A514M", _PLATE,
             note="quenched and tempered; Fy 90 ksi for t > 2-1/2 in.")

# ---------------------------------------------------------------------------
# Hollow structural sections -- Sect. A3.1a(b), p. 16.1-7
# A500 strengths differ between round and rectangular product, which is a
# frequent source of error, so the two are separate grades rather than a flag.
# ---------------------------------------------------------------------------
A500_B_RECT = Steel("A500 Gr. B (rect.)", 46.0, 58.0, "ASTM A500/A500M", _HSS)
A500_B_ROUND = Steel("A500 Gr. B (round)", 42.0, 58.0, "ASTM A500/A500M", _HSS)
A500_C_RECT = Steel("A500 Gr. C (rect.)", 50.0, 62.0, "ASTM A500/A500M", _HSS)
A500_C_ROUND = Steel("A500 Gr. C (round)", 46.0, 62.0, "ASTM A500/A500M", _HSS)
A501_A = Steel("A501 Gr. A", 36.0, 58.0, "ASTM A501/A501M", _HSS)
A501_B = Steel("A501 Gr. B", 50.0, 70.0, "ASTM A501/A501M", _HSS)
A847 = Steel("A847", 50.0, 70.0, "ASTM A847/A847M", _HSS, note="weathering HSS")
A1065_50 = Steel("A1065 Gr. 50", 50.0, 60.0, "ASTM A1065/A1065M", _HSS,
                 note="full nominal wall thickness applies (Sect. B4.2)")
A1085 = Steel("A1085", 50.0, 65.0, "ASTM A1085/A1085M", _HSS,
              note="full nominal wall thickness applies (Sect. B4.2)")
A53_B = Steel("A53 Gr. B", 35.0, 60.0, "ASTM A53/A53M Grade B", _HSS,
              note="pipe; designed as round HSS per the User Note to Sect. B4.2")


#: Every grade, keyed by its designation. Lookups go through :func:`grade`.
GRADES: dict[str, Steel] = {
    s.name: s
    for s in (
        A36, A529_50, A529_55,
        A572_42, A572_50, A572_55, A572_60, A572_65,
        A588, A709_36, A709_50,
        A913_50, A913_60, A913_65, A913_70,
        A992, A1043_36, A1043_50,
        A242, A283_C, A514,
        A500_B_RECT, A500_B_ROUND, A500_C_RECT, A500_C_ROUND,
        A501_A, A501_B, A847, A1065_50, A1085, A53_B,
    )
}

#: Spellings people actually type, mapped onto canonical names.
_ALIASES: dict[str, str] = {
    "a36": "A36",
    "a992": "A992",
    "a992/a992m": "A992",
    "a572gr50": "A572 Gr. 50",
    "a572-50": "A572 Gr. 50",
    "a572 grade 50": "A572 Gr. 50",
    "a500b": "A500 Gr. B (rect.)",
    "a500 gr b": "A500 Gr. B (rect.)",
    "a500c": "A500 Gr. C (rect.)",
    "a500 gr c": "A500 Gr. C (rect.)",
    "a1085": "A1085",
    "a53": "A53 Gr. B",
    "s355": "A572 Gr. 50",  # nearest AISC-approved equivalent; see note below
}


def grade(name: str) -> Steel:
    """Look a grade up by designation, tolerantly.

    Matching ignores case, spaces, dots and hyphens, so ``"A572 Gr. 50"``,
    ``"a572-50"`` and ``"A572GR50"`` all resolve.

    A European designation such as ``"S355"`` maps to its nearest AISC-approved
    equivalent, which is an *engineering judgement, not an equivalence* -- EN
    10025 grades are not listed in Sect. A3.1a. Check the mill certificate
    before relying on it.

    Raises
    ------
    MaterialError
        If the designation is not recognised.
    """
    if name in GRADES:
        return GRADES[name]

    key = name.strip().lower()
    if key in _ALIASES:
        return GRADES[_ALIASES[key]]

    squashed = "".join(c for c in key if c.isalnum())
    for canonical, steel in GRADES.items():
        if "".join(c for c in canonical.lower() if c.isalnum()) == squashed:
            return steel
    for alias, canonical in _ALIASES.items():
        if "".join(c for c in alias if c.isalnum()) == squashed:
            return GRADES[canonical]

    raise MaterialError(
        f"unknown steel grade {name!r}. Known grades: {', '.join(sorted(GRADES))}"
    )


def grades_for(form: ProductForm) -> tuple[Steel, ...]:
    """Every grade approved for a product form, per Sect. A3.1a, p. 16.1-7."""
    return tuple(s for s in GRADES.values() if form in s.forms)


def design_wall_thickness(nominal_t: Inch, steel: Steel, *, box_section: bool = False) -> Inch:
    """Design wall thickness ``t`` for an HSS.

    AISC 360-16, Sect. B4.2, p. 16.1-20.

    Equal to the nominal thickness for box sections and for HSS produced to
    ASTM A1065 or A1085; **0.93 times nominal** for HSS produced to any other
    approved standard, which covers the A500 material most HSS is supplied in.

    Missing this factor overstates the area of a typical A500 HSS by about 7.5%,
    so it is applied here rather than left to the caller.

    Parameters
    ----------
    nominal_t:
        Nominal wall thickness, in.
    steel:
        The HSS grade -- determines which rule applies.
    box_section:
        Set for a built-up box section, which always uses the full thickness.

    Raises
    ------
    MaterialError
        If ``steel`` is not an HSS grade, or the thickness is not positive.
    """
    if nominal_t <= 0.0:
        raise MaterialError(f"nominal wall thickness must be positive, got {nominal_t}")
    if not box_section:
        steel.require_form(ProductForm.HSS)
    if box_section or steel.standard in _FULL_THICKNESS_HSS:
        return nominal_t
    return 0.93 * nominal_t
