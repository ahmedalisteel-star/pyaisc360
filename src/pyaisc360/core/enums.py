"""Enumerations shared across chapters: limit states, slenderness, geometry.

The limit-state names follow the abbreviations AISC uses in the selection
tables (Table User Note E1.1, p. 16.1-34 and Table User Note F1.1, p. 16.1-45)
so a report reads the same way the Specification does.
"""

from __future__ import annotations

from enum import Enum

__all__ = [
    "LimitState",
    "AxialSlenderness",
    "FlexuralSlenderness",
    "ElementSupport",
    "Axis",
    "ProductForm",
    "StabilityMethod",
]


class LimitState(str, Enum):
    """Named limit states, grouped by the chapter that defines them.

    ``str`` mixin so a limit state serialises straight into JSON/IFC property
    sets and compares equal to its own name in reports.
    """

    # -- Chapter D, tension ------------------------------------------------
    TENSILE_YIELDING = "tensile yielding"
    TENSILE_RUPTURE = "tensile rupture"
    SHEAR_RUPTURE_PIN = "shear rupture on effective area (pin)"
    BEARING_PIN = "bearing on projected area (pin)"

    # -- Chapter E, compression -------------------------------------------
    FLEXURAL_BUCKLING = "flexural buckling"
    TORSIONAL_BUCKLING = "torsional buckling"
    FLEXURAL_TORSIONAL_BUCKLING = "flexural-torsional buckling"
    LOCAL_BUCKLING = "local buckling"

    # -- Chapter F, flexure ------------------------------------------------
    YIELDING = "yielding"
    PLASTIC_MOMENT = "yielding (plastic moment)"
    COMPRESSION_FLANGE_YIELDING = "compression flange yielding"
    LATERAL_TORSIONAL_BUCKLING = "lateral-torsional buckling"
    FLANGE_LOCAL_BUCKLING = "compression flange local buckling"
    WEB_LOCAL_BUCKLING = "web local buckling"
    TENSION_FLANGE_YIELDING = "tension flange yielding"
    LEG_LOCAL_BUCKLING = "leg local buckling"
    FLANGE_RUPTURE = "tensile rupture of the flange"

    # -- Chapter G, shear --------------------------------------------------
    SHEAR_YIELDING = "shear yielding"
    SHEAR_BUCKLING = "shear buckling"
    SHEAR_RUPTURE = "shear rupture"
    TENSION_FIELD_ACTION = "shear with tension field action"

    # -- Chapter H, combined forces ---------------------------------------
    COMBINED_AXIAL_FLEXURE = "combined axial force and flexure"
    COMBINED_TORSION = "combined torsion, flexure, shear and/or axial force"
    TORSIONAL_YIELDING = "torsional yielding"
    TORSIONAL_BUCKLING_HSS = "torsional buckling of HSS"

    # -- Chapter J, connections -------------------------------------------
    BLOCK_SHEAR = "block shear rupture"
    BOLT_SHEAR = "bolt shear rupture"
    BOLT_TENSION = "bolt tensile rupture"
    BOLT_BEARING = "bearing at bolt holes"
    BOLT_TEAROUT = "tearout at bolt holes"
    SLIP = "slip"
    WELD_RUPTURE = "weld rupture"
    BASE_METAL_RUPTURE = "base metal rupture"
    WEB_LOCAL_YIELDING = "web local yielding"
    WEB_LOCAL_CRIPPLING = "web local crippling"
    WEB_COMPRESSION_BUCKLING = "web compression buckling"
    WEB_SIDESWAY_BUCKLING = "web sidesway buckling"
    FLANGE_LOCAL_BENDING = "flange local bending"
    CONCRETE_BEARING = "bearing on concrete"


class AxialSlenderness(str, Enum):
    """Classification for members in axial compression.

    Section B4.1 recognises only two classes here: an element is nonslender if
    its width-to-thickness ratio does not exceed lambda_r from Table B4.1a,
    otherwise it is slender.  There is no "compact" for axial compression --
    conflating this with the flexural classification is a classic error, so the
    two enums are deliberately not interchangeable.

    AISC 360-16, Sect. B4.1, p. 16.1-16.
    """

    NONSLENDER = "nonslender"
    SLENDER = "slender"


class FlexuralSlenderness(str, Enum):
    """Classification for members subject to flexure, per Table B4.1b.

    AISC 360-16, Sect. B4.1, p. 16.1-16.
    """

    COMPACT = "compact"
    NONCOMPACT = "noncompact"
    SLENDER = "slender"


class ElementSupport(str, Enum):
    """Whether a compression element is supported along one edge or two.

    Sect. B4.1a (unstiffened, one edge) and B4.1b (stiffened, two edges),
    p. 16.1-16.
    """

    UNSTIFFENED = "unstiffened"
    STIFFENED = "stiffened"


class Axis(str, Enum):
    """Bending or buckling axis."""

    MAJOR = "x"
    MINOR = "y"


class ProductForm(str, Enum):
    """Product form, per the ASTM groupings of Sect. A3.1a, p. 16.1-7.

    Governs which grades are permitted and, for HSS, which design wall
    thickness applies (Sect. B4.2, p. 16.1-19).
    """

    SHAPE = "hot-rolled structural shape"
    HSS = "hollow structural section"
    PLATE = "plate"
    BAR = "bar"
    SHEET = "sheet"
    ROD = "rod"


class StabilityMethod(str, Enum):
    """Which method of design for stability is in use.

    Chapter C's direct analysis method is permitted for all structures; the two
    Appendix 7 alternatives carry limitations. The choice changes what the
    analysis must do, not what the member equations are -- Sect. C3 and Sects.
    7.2.3/7.3.3 all route the available strength through Chapters D-K unchanged.

    AISC 360-16, Sects. C1.1 and C1.2, p. 16.1-22.
    """

    #: Sect. C1.1 -- reduced stiffness, notional loads, Lc = L. No limitations.
    DIRECT_ANALYSIS = "direct analysis method (Chapter C)"
    #: Appendix 7.2 -- nominal stiffness, stability carried through K.
    EFFECTIVE_LENGTH = "effective length method (Appendix 7.2)"
    #: Appendix 7.3 -- first-order analysis with an additional lateral load.
    FIRST_ORDER = "first-order analysis method (Appendix 7.3)"
