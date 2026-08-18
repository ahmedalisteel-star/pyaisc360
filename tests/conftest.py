"""Shared fixtures.

Section properties used here are the published AISC *Shapes Database* v15.0
values for the shapes that appear in the Design Examples. They are test data,
not a shipped catalog -- pyaisc360 deliberately has no catalog of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pyaisc360.sections import SectionAdapter  # noqa: E402

#: W18x50, ASTM A6 -- the workhorse of the Chapter F Design Examples.
W18X50_PROPS: dict[str, float] = {
    "Ag": 14.7,
    "d": 18.0,
    "bf": 7.50,
    "tf": 0.570,
    "tw": 0.355,
    "kdes": 0.972,
    "Ix": 800.0,
    "Iy": 40.1,
    "Sx": 88.9,
    "Sy": 10.7,
    "Zx": 101.0,
    "Zy": 16.6,
    "rx": 7.38,
    "ry": 1.65,
    "J": 1.24,
    "Cw": 3040.0,
    "rts": 1.98,
    "ho": 17.4,
    "bf_2tf": 6.57,
    "h_tw": 45.2,
}

#: W14x90 -- used by the Chapter E compression examples.
W14X90_PROPS: dict[str, float] = {
    "Ag": 26.5,
    "d": 14.0,
    "bf": 14.5,
    "tf": 0.710,
    "tw": 0.440,
    "Ix": 999.0,
    "Iy": 362.0,
    "Sx": 143.0,
    "Sy": 49.9,
    "Zx": 157.0,
    "Zy": 75.6,
    "rx": 6.14,
    "ry": 3.70,
    "J": 4.06,
    "Cw": 16000.0,
    "rts": 4.10,
    "ho": 13.3,
    "bf_2tf": 10.2,
    "h_tw": 25.9,
}


@pytest.fixture
def w18x50_props() -> dict[str, float]:
    return dict(W18X50_PROPS)


@pytest.fixture
def w18x50() -> SectionAdapter:
    return SectionAdapter.from_mapping({"name": "W18X50", **W18X50_PROPS})


@pytest.fixture
def w14x90() -> SectionAdapter:
    return SectionAdapter.from_mapping({"name": "W14X90", **W14X90_PROPS})


@pytest.fixture
def w18x50_metric() -> SectionAdapter:
    """The same shape expressed in millimetres, to exercise conversion."""
    mm = 25.4
    return SectionAdapter.from_mapping(
        {
            "name": "W18X50 (mm)",
            "Ag": 14.7 * mm**2,
            "d": 18.0 * mm,
            "tf": 0.570 * mm,
            "bf": 7.50 * mm,
            "Iy": 40.1 * mm**4,
            "Cw": 3040.0 * mm**6,
            "Sx": 88.9 * mm**3,
        },
        metric=True,
    )
