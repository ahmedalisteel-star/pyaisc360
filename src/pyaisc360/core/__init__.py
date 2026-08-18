"""Cross-cutting infrastructure: units, configuration, results, citations."""

from __future__ import annotations

from .citations import Citation, cite, known_equations, spec_index
from .config import DEFAULT_SETTINGS, Basis, DesignSettings
from .enums import (
    AxialSlenderness,
    Axis,
    ElementSupport,
    FlexuralSlenderness,
    LimitState,
    ProductForm,
    StabilityMethod,
)
from .exceptions import (
    AISC360Error,
    ConvergenceError,
    GeometryError,
    MaterialError,
    MissingPropertyError,
    OutOfScopeError,
)
from .result import LimitStateResult, StrengthResult

__all__ = [
    "Citation",
    "cite",
    "known_equations",
    "spec_index",
    "Basis",
    "DesignSettings",
    "DEFAULT_SETTINGS",
    "LimitState",
    "AxialSlenderness",
    "FlexuralSlenderness",
    "ElementSupport",
    "Axis",
    "ProductForm",
    "StabilityMethod",
    "AISC360Error",
    "OutOfScopeError",
    "GeometryError",
    "MissingPropertyError",
    "MaterialError",
    "ConvergenceError",
    "LimitStateResult",
    "StrengthResult",
]
