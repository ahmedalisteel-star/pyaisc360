"""Design philosophy selection and library-wide settings.

Section B3.1 (LRFD) and B3.2 (ASD) are two ways of comparing the *same* nominal
strength against demand.  The Specification never gives a separate nominal
strength for ASD -- only a different factor.  So the library computes Rn once
and applies phi or Omega at the very end; there is no forked ASD code path to
drift out of sync.

AISC 360-16, Sect. B3.1-B3.2, p. 16.1-12.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .enums import StabilityMethod

__all__ = ["Basis", "DesignSettings", "DEFAULT_SETTINGS"]


class Basis(str, Enum):
    """Design philosophy.

    LRFD -- Ru <= phi*Rn      (Eq. B3-1)
    ASD  -- Ra <= Rn/Omega    (Eq. B3-2)
    """

    LRFD = "LRFD"
    ASD = "ASD"

    @property
    def is_lrfd(self) -> bool:
        return self is Basis.LRFD

    @property
    def demand_symbol(self) -> str:
        """``Ru`` for LRFD, ``Ra`` for ASD -- as the Specification labels them."""
        return "Ru" if self is Basis.LRFD else "Ra"


@dataclass(frozen=True, slots=True)
class DesignSettings:
    """Library-wide behaviour switches.

    Attributes
    ----------
    basis:
        LRFD or ASD.  Applied only at the final available-strength step.
    strict:
        When True (default), a provision invoked outside its stated range of
        applicability raises :class:`~pyaisc360.core.exceptions.OutOfScopeError`
        instead of extrapolating.  Turning this off is a deliberate act by an
        engineer who has justified the extension; it is never the default.
    check_factor_calibration:
        Verify that phi and Omega are mutually consistent (Omega ~ 1.5/phi, the
        calibration the Specification uses throughout).  Catches a transposed
        factor pair, which is otherwise invisible because both numbers are
        individually plausible.
    cb_cantilever_unity:
        For cantilevers with warping prevented at the support and a free end
        unbraced, Sect. F1(c) requires Cb = 1.0 rather than Eq. F1-1.
    stability_method:
        Which method of design for stability the analysis used. Under
        ``DIRECT_ANALYSIS`` the required strengths must come from an analysis
        with the Sect. C2.3 stiffness reductions applied, and Sect. C3 fixes the
        effective length at the unbraced length. The member equations of
        Chapters D-K are the same under all three -- the flag records what the
        *analysis* did, so a report can state it and so
        :func:`pyaisc360.chapter_c.effective_length_factor` can enforce Sect. C3.
    """

    basis: Basis = Basis.LRFD
    strict: bool = True
    check_factor_calibration: bool = True
    cb_cantilever_unity: bool = True
    stability_method: StabilityMethod = StabilityMethod.DIRECT_ANALYSIS

    def with_basis(self, basis: Basis) -> DesignSettings:
        """Return a copy switched to the other design philosophy."""
        return DesignSettings(
            basis=basis,
            strict=self.strict,
            check_factor_calibration=self.check_factor_calibration,
            cb_cantilever_unity=self.cb_cantilever_unity,
            stability_method=self.stability_method,
        )


DEFAULT_SETTINGS = DesignSettings()
