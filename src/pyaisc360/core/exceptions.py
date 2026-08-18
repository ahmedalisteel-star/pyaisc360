"""Exception hierarchy.

A design library must fail loudly. Every one of these represents a case where
returning a number would be worse than raising: the provision does not apply,
the geometry is impossible, or a property the equation needs is missing. None of
them are recoverable by extrapolating -- that is how an unconservative answer
gets shipped.
"""

from __future__ import annotations

__all__ = [
    "AISC360Error",
    "OutOfScopeError",
    "GeometryError",
    "MissingPropertyError",
    "MaterialError",
    "ConvergenceError",
]


class AISC360Error(Exception):
    """Base class for every error raised by pyaisc360."""


class OutOfScopeError(AISC360Error):
    """The member or loading falls outside the provision's stated applicability.

    Raised rather than extrapolating a fitted curve past its validated range --
    e.g. applying Section F7 (rectangular HSS) to a section whose flange is
    slender when the invoked branch assumes compact.
    """


class GeometryError(AISC360Error):
    """A cross-section or unbraced length is physically impossible or inconsistent."""


class MissingPropertyError(AISC360Error, AttributeError):
    """A required section property is absent and cannot be derived.

    Subclasses :class:`AttributeError` so that ``getattr(section, "Cw", None)``
    and ``hasattr`` keep working against adapted third-party section objects.
    """


class MaterialError(AISC360Error):
    """The steel grade is unsuitable for the requested product form or provision."""


class ConvergenceError(AISC360Error):
    """An iterative solution (e.g. the Eq. E4-4 cubic) failed to converge."""
