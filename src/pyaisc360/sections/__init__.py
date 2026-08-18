"""Section property interface.

No catalog ships with pyaisc360 -- see :mod:`pyaisc360.sections.base` for why,
and for how to plug an existing section library in.
"""

from __future__ import annotations

from .base import (
    LENGTH_EXPONENT,
    AngleSection,
    ChannelSection,
    IShapedSection,
    RectangularHSS,
    RoundHSS,
    SectionAdapter,
    SectionProperties,
    TeeSection,
    available_properties,
    derive_c_channel,
    derive_ho,
    derive_rts,
    require_properties,
)

__all__ = [
    "SectionProperties",
    "IShapedSection",
    "ChannelSection",
    "TeeSection",
    "AngleSection",
    "RectangularHSS",
    "RoundHSS",
    "SectionAdapter",
    "LENGTH_EXPONENT",
    "require_properties",
    "available_properties",
    "derive_rts",
    "derive_ho",
    "derive_c_channel",
]
