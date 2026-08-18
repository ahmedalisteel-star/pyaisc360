"""pyaisc360 -- design verification to ANSI/AISC 360-16.

Wave 0 (foundations) is implemented: units, configuration, results, citations,
shared mechanics, materials and the section-property interface. The chapter
modules follow the order in ``ROADMAP.md``.

    >>> from pyaisc360 import A992, Basis, utils
    >>> Fe = utils.elastic_buckling_stress(80.0, A992.E)
    >>> round(utils.flexural_buckling_stress(A992.Fy, Fe), 3)
    31.314
"""

from __future__ import annotations

from . import materials, utils
from .core import (
    DEFAULT_SETTINGS,
    AISC360Error,
    AxialSlenderness,
    Axis,
    Basis,
    Citation,
    DesignSettings,
    ElementSupport,
    FlexuralSlenderness,
    GeometryError,
    LimitState,
    LimitStateResult,
    MaterialError,
    MissingPropertyError,
    OutOfScopeError,
    ProductForm,
    StrengthResult,
    cite,
)
from .materials import (
    A36,
    A500_B_RECT,
    A500_B_ROUND,
    A500_C_RECT,
    A500_C_ROUND,
    A572_50,
    A913_65,
    A992,
    A1085,
    E_STEEL,
    G_STEEL,
    Steel,
    design_wall_thickness,
    grade,
)
from .sections import SectionAdapter, require_properties

__version__ = "0.1.0.dev0"

#: The edition this library implements. Every citation resolves against it.
SPECIFICATION = "ANSI/AISC 360-16, Specification for Structural Steel Buildings (July 7, 2016)"

__all__ = [
    "__version__",
    "SPECIFICATION",
    # configuration
    "Basis",
    "DesignSettings",
    "DEFAULT_SETTINGS",
    # results and traceability
    "StrengthResult",
    "LimitStateResult",
    "LimitState",
    "Citation",
    "cite",
    # classification
    "AxialSlenderness",
    "FlexuralSlenderness",
    "ElementSupport",
    "Axis",
    "ProductForm",
    # materials
    "Steel",
    "grade",
    "design_wall_thickness",
    "E_STEEL",
    "G_STEEL",
    "A36",
    "A992",
    "A572_50",
    "A913_65",
    "A500_B_RECT",
    "A500_B_ROUND",
    "A500_C_RECT",
    "A500_C_ROUND",
    "A1085",
    # sections
    "SectionAdapter",
    "require_properties",
    # errors
    "AISC360Error",
    "OutOfScopeError",
    "GeometryError",
    "MissingPropertyError",
    "MaterialError",
    # modules
    "utils",
    "materials",
]
