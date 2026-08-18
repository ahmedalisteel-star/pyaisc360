"""Section property interface -- protocols and an adapter, no catalog data.

pyaisc360 deliberately ships **no shapes database**. The AISC *Shapes Database*
is a separate publication, and most users already have a section library of
their own (a BIM model, an EN profile table, a fabricator's stock list). What
the chapters need is not a catalog but a *contract*: given some object, can I
read ``Ag``, ``Zx``, ``rts``, ``J``?

This module defines that contract three ways:

* :class:`SectionProperties` and its per-shape refinements -- structural-typing
  protocols. Any object already exposing the AISC property names satisfies them
  with no inheritance and no registration.
* :class:`SectionAdapter` -- wraps an object whose names differ, resolving
  aliases and, if the source is metric, converting to the US customary units the
  Specification's coefficients require.
* Derivation helpers -- ``rts`` and ``c`` computed from more primitive
  properties when a source does not tabulate them.

Canonical names follow AISC *Shapes Database* v15.0 nomenclature, with one
deliberate exception: gross area is ``Ag``, as the Specification writes it,
rather than the database's bare ``A``. ``A`` is accepted as an alias.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from ..core.exceptions import GeometryError, MissingPropertyError
from ..core.units import MM_PER_IN, Inch, Inch2, Inch3, Inch4, Inch6, Ratio

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


# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------
@runtime_checkable
class SectionProperties(Protocol):
    """The minimum every member check needs: a name and a gross area.

    ``Ag`` alone serves Chapter D tension yielding and Chapter E once a critical
    stress is known. Everything richer is a refinement below.
    """

    name: str
    Ag: Inch2


@runtime_checkable
class IShapedSection(Protocol):
    """W, S, M and HP shapes, and built-up plate girders.

    Consumed by Sects. E3/E4/E7, F2-F5, G2 and H1.

    ``rts`` (Eq. F2-7) and ``ho`` are listed because the Shapes Database
    tabulates them; :class:`SectionAdapter` derives both when a source does not.
    """

    name: str
    Ag: Inch2
    d: Inch
    bf: Inch
    tf: Inch
    tw: Inch
    Ix: Inch4
    Iy: Inch4
    Sx: Inch3
    Sy: Inch3
    Zx: Inch3
    Zy: Inch3
    rx: Inch
    ry: Inch
    J: Inch4
    Cw: Inch6
    rts: Inch
    ho: Inch


@runtime_checkable
class ChannelSection(Protocol):
    """C and MC shapes. Sect. F2 applies, but with ``c`` from Eq. F2-8b."""

    name: str
    Ag: Inch2
    d: Inch
    bf: Inch
    tf: Inch
    tw: Inch
    Ix: Inch4
    Iy: Inch4
    Sx: Inch3
    Zx: Inch3
    rx: Inch
    ry: Inch
    J: Inch4
    Cw: Inch6
    ho: Inch
    eo: Inch
    """Horizontal distance from the web's outer face to the shear centre, in."""


@runtime_checkable
class TeeSection(Protocol):
    """WT, MT and ST shapes. Sects. E4, F9, G3."""

    name: str
    Ag: Inch2
    d: Inch
    bf: Inch
    tf: Inch
    tw: Inch
    Ix: Inch4
    Iy: Inch4
    Sx: Inch3
    Zx: Inch3
    rx: Inch
    ry: Inch
    J: Inch4
    y: Inch
    """Distance from the flange face to the elastic neutral axis, in."""


@runtime_checkable
class AngleSection(Protocol):
    """Single angles. Sects. E5, F10, G3.

    ``z`` is the principal minor axis, which for an unequal-leg angle is not
    parallel to either leg -- Sect. F10 works in principal axes throughout.
    """

    name: str
    Ag: Inch2
    b: Inch
    d: Inch
    t: Inch
    Ix: Inch4
    Iy: Inch4
    Sx: Inch3
    Sy: Inch3
    Zx: Inch3
    rx: Inch
    ry: Inch
    rz: Inch
    J: Inch4
    x: Inch
    y: Inch


@runtime_checkable
class RectangularHSS(Protocol):
    """Square and rectangular HSS and box sections. Sects. E7, F7, G4, H1, K.

    ``t`` is the **design** wall thickness of Sect. B4.2 (p. 16.1-20), i.e.
    0.93 x nominal for A500 material -- see
    :func:`pyaisc360.materials.design_wall_thickness`.
    """

    name: str
    Ag: Inch2
    Ht: Inch
    B: Inch
    t: Inch
    Ix: Inch4
    Iy: Inch4
    Sx: Inch3
    Sy: Inch3
    Zx: Inch3
    Zy: Inch3
    rx: Inch
    ry: Inch
    J: Inch4
    C: Inch3
    """HSS torsional constant, in.^3 -- used by Sect. H3.1."""


@runtime_checkable
class RoundHSS(Protocol):
    """Round HSS and pipe. Sects. E7, F8, G5, H3."""

    name: str
    Ag: Inch2
    D: Inch
    t: Inch
    Ix: Inch4
    Sx: Inch3
    Zx: Inch3
    rx: Inch
    J: Inch4
    C: Inch3


# ---------------------------------------------------------------------------
# Dimensional bookkeeping for unit conversion
# ---------------------------------------------------------------------------
#: Length exponent of each canonical property, used to convert a metric source
#: to inches. Ratios and dimensionless constants carry 0.
LENGTH_EXPONENT: dict[str, int] = {
    # areas
    "Ag": 2, "An": 2, "Ae": 2, "Aw": 2, "Afg": 2, "Afn": 2,
    # lengths
    "d": 1, "bf": 1, "tf": 1, "tw": 1, "t": 1, "tnom": 1, "b": 1, "h": 1,
    "B": 1, "Ht": 1, "D": 1, "kdes": 1, "kdet": 1, "k1": 1, "ho": 1, "eo": 1,
    "x": 1, "y": 1, "xo": 1, "yo": 1, "ro": 1,
    "rx": 1, "ry": 1, "rz": 1, "rts": 1,
    # section moduli and the HSS torsional constant
    "Sx": 3, "Sy": 3, "Sz": 3, "Zx": 3, "Zy": 3, "Zz": 3, "C": 3,
    # second moments and the torsional constant
    "Ix": 4, "Iy": 4, "Iz": 4, "J": 4,
    # warping constant
    "Cw": 6,
    # dimensionless
    "bf_2tf": 0, "h_tw": 0, "b_t": 0, "h_t": 0, "D_t": 0, "d_t": 0,
    "H": 0, "tan_alpha": 0, "c": 0,
}

#: Alternative spellings accepted for each canonical property. Kept deliberately
#: tight: a wrong guess here silently feeds the wrong number into a design
#: equation, so anything ambiguous must be given explicitly via ``overrides``.
_ALIASES: dict[str, tuple[str, ...]] = {
    "Ag": ("Ag", "A", "area", "gross_area", "AREA"),
    "d": ("d", "depth", "H_depth"),
    "bf": ("bf", "b_f", "flange_width", "bF"),
    "tf": ("tf", "t_f", "flange_thickness"),
    "tw": ("tw", "t_w", "web_thickness"),
    "t": ("t", "tdes", "t_des", "design_thickness", "thickness"),
    "tnom": ("tnom", "t_nom", "nominal_thickness"),
    "Ix": ("Ix", "I_x", "Iy_major", "Iyy"),
    "Iy": ("Iy", "I_y", "Izz"),
    "Sx": ("Sx", "S_x", "Wx", "Wel_y"),
    "Sy": ("Sy", "S_y", "Wy", "Wel_z"),
    "Zx": ("Zx", "Z_x", "Wpl_y"),
    "Zy": ("Zy", "Z_y", "Wpl_z"),
    "rx": ("rx", "r_x", "iy_radius"),
    "ry": ("ry", "r_y"),
    "rz": ("rz", "r_z"),
    "rts": ("rts", "r_ts"),
    "J": ("J", "It", "torsion_constant"),
    "Cw": ("Cw", "C_w", "Iw", "warping_constant"),
    "ho": ("ho", "h_o", "flange_centroid_distance"),
    "B": ("B", "width"),
    "Ht": ("Ht", "H", "height"),
    "D": ("D", "OD", "outside_diameter"),
    "C": ("C", "HSS_C", "torsional_modulus"),
    "name": ("name", "designation", "label", "profile", "Name"),
}


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------
class SectionAdapter:
    """Expose any section object under the canonical AISC property names.

    Resolution order for a property, first hit wins:

    1. an explicit entry in ``overrides``
    2. the canonical name on the source object
    3. a known alias on the source object
    4. a derivation, for the small set that can be derived (``rts``, ``ho``, ``c``)

    Anything unresolved raises :class:`MissingPropertyError` -- the adapter never
    substitutes a default, because a plausible wrong section property is far more
    dangerous than a crash.

    Parameters
    ----------
    source:
        Any object or mapping carrying section properties.
    metric:
        Set when the source is in millimetres. Each property is then converted
        by the appropriate power of 25.4 on read (see :data:`LENGTH_EXPONENT`).
        This is the usual case for a FreeCAD model or an EN profile table.
    overrides:
        Canonical name -> attribute/key name on the source, for anything the
        aliases do not cover.
    name:
        Section designation, when the source has none.

    Examples
    --------
    An EN profile record in millimetres, with its own naming::

        sec = SectionAdapter(ipe300, metric=True,
                             overrides={"Ag": "A_mm2", "Zx": "Wpl_y"})
        sec.Ag       # in.^2, converted
        sec.rts      # derived from Iy, Cw and Sx per Eq. F2-7
    """

    __slots__ = ("_source", "_metric", "_overrides", "_name", "_cache")

    def __init__(
        self,
        source: Any,
        *,
        metric: bool = False,
        overrides: Mapping[str, str] | None = None,
        name: str | None = None,
    ) -> None:
        self._source = source
        self._metric = metric
        self._overrides = dict(overrides or {})
        self._name = name
        self._cache: dict[str, Any] = {}

    # -- construction -----------------------------------------------------
    @classmethod
    def from_mapping(
        cls, properties: Mapping[str, float | str], *, metric: bool = False
    ) -> SectionAdapter:
        """Adapt a plain dict of properties -- handy for tests and one-offs."""
        return cls(dict(properties), metric=metric)

    # -- reading ----------------------------------------------------------
    def _raw(self, canonical: str) -> Any:
        """Fetch a property from the source without converting or deriving."""
        candidates: list[str] = []
        if canonical in self._overrides:
            candidates.append(self._overrides[canonical])
        candidates.append(canonical)
        candidates.extend(a for a in _ALIASES.get(canonical, ()) if a != canonical)

        src = self._source
        if isinstance(src, Mapping):
            for key in candidates:
                if key in src:
                    return src[key]
            return None
        for key in candidates:
            value = getattr(src, key, None)
            if value is not None:
                return value
        return None

    def __getattr__(self, canonical: str) -> Any:
        if canonical.startswith("_"):
            raise AttributeError(canonical)
        if canonical in self._cache:
            return self._cache[canonical]

        if canonical == "name":
            value: Any = self._name or self._raw("name") or repr(self._source)
            self._cache["name"] = value
            return value

        raw = self._raw(canonical)
        if raw is None:
            raw = self._derive(canonical)
        if raw is None:
            raise MissingPropertyError(
                f"section {self.name!r} has no property {canonical!r} and it cannot "
                f"be derived. Supply it via overrides={{'{canonical}': '<source name>'}} "
                "or add it to the source object."
            )

        value = self._convert(canonical, float(raw))
        self._cache[canonical] = value
        return value

    def _convert(self, canonical: str, value: float) -> float:
        if not self._metric:
            return value
        exponent = LENGTH_EXPONENT.get(canonical)
        if exponent is None:
            raise MissingPropertyError(
                f"cannot convert {canonical!r} from metric: its dimension is unknown. "
                "Add it to LENGTH_EXPONENT, or pass the section in US customary units."
            )
        return value / MM_PER_IN**exponent if exponent else value

    def _derive(self, canonical: str) -> float | None:
        """Compute the few properties that follow from more primitive ones.

        Derived values are computed in the *source's* units, so the normal
        conversion applies afterwards -- which is why ``rts`` is derived from
        raw ``Iy``/``Cw``/``Sx`` rather than from the converted properties.
        """
        if canonical == "ho":
            d, tf = self._raw("d"), self._raw("tf")
            if d is not None and tf is not None:
                return derive_ho(float(d), float(tf))
        elif canonical == "rts":
            Iy, Cw, Sx = self._raw("Iy"), self._raw("Cw"), self._raw("Sx")
            if Iy is not None and Cw is not None and Sx is not None:
                return derive_rts(float(Iy), float(Cw), float(Sx))
        elif canonical == "bf_2tf":
            bf, tf = self._raw("bf"), self._raw("tf")
            if bf is not None and tf is not None and float(tf) > 0.0:
                return float(bf) / (2.0 * float(tf))
        return None

    # -- introspection ----------------------------------------------------
    def has(self, *canonical: str) -> bool:
        """True when every named property is readable (directly or derived)."""
        for prop in canonical:
            try:
                getattr(self, prop)
            except MissingPropertyError:
                return False
        return True

    def to_dict(self, *canonical: str) -> dict[str, float]:
        """Read several properties at once, in US customary units."""
        return {prop: getattr(self, prop) for prop in canonical}

    def __repr__(self) -> str:
        units = "mm" if self._metric else "in."
        return f"<SectionAdapter {self.name!r} source={units}>"


# ---------------------------------------------------------------------------
# Derivations and checks
# ---------------------------------------------------------------------------
def derive_ho(d: Inch, tf: Inch) -> Inch:
    """Distance between flange centroids, ``ho = d - tf``.

    Geometry, not a numbered equation -- ``ho`` is defined in the ``where``
    clause of Sect. F2, p. 16.1-48. Exact for a doubly symmetric I-shape with
    parallel flanges; for a tapered-flange shape (S, MC) use the tabulated value,
    since ``tf`` is then the mean thickness (Sect. B4.1b, p. 16.1-20).
    """
    if d <= 0.0 or tf <= 0.0:
        raise GeometryError(f"d and tf must be positive, got d={d}, tf={tf}")
    if tf >= d / 2.0:
        raise GeometryError(f"flange thickness {tf} is impossible for a depth of {d}")
    return d - tf


def derive_rts(Iy: Inch4, Cw: Inch6, Sx: Inch3) -> Inch:
    """Effective radius of gyration for lateral-torsional buckling, ``rts``.

    AISC 360-16, Eq. F2-7, Sect. F2.2, p. 16.1-48::

        rts^2 = sqrt(Iy * Cw) / Sx

    Use the tabulated value where one exists. Deriving it is exact only for a
    section whose ``Cw`` is itself exact, which is why this is a fallback rather
    than the primary path.
    """
    if Iy <= 0.0 or Cw <= 0.0 or Sx <= 0.0:
        raise GeometryError(f"Iy, Cw and Sx must be positive, got {Iy}, {Cw}, {Sx}")
    return math.sqrt(math.sqrt(Iy * Cw) / Sx)


def derive_c_channel(ho: Inch, Iy: Inch4, Cw: Inch6) -> Ratio:
    """Coefficient ``c`` for a channel.

    AISC 360-16, Eq. F2-8b, Sect. F2.2, p. 16.1-48::

        c = (ho/2) * sqrt(Iy/Cw)

    For a doubly symmetric I-shape ``c = 1`` (Eq. F2-8a); no helper is needed for
    that case.
    """
    if ho <= 0.0 or Iy <= 0.0 or Cw <= 0.0:
        raise GeometryError(f"ho, Iy and Cw must be positive, got {ho}, {Iy}, {Cw}")
    return (ho / 2.0) * math.sqrt(Iy / Cw)


def require_properties(section: Any, *names: str) -> dict[str, float]:
    """Read the named properties or raise, listing everything that is missing.

    Chapters call this on entry so a member with an incomplete property set
    fails once with a complete list, rather than repeatedly one attribute at a
    time as the calculation unwinds.

    Raises
    ------
    MissingPropertyError
        Naming every property that could not be read.
    """
    values: dict[str, float] = {}
    missing: list[str] = []
    for name in names:
        try:
            values[name] = float(getattr(section, name))
        except (AttributeError, TypeError, ValueError):
            missing.append(name)
    if missing:
        label = getattr(section, "name", section)
        raise MissingPropertyError(
            f"section {label!r} is missing required properties: {', '.join(missing)}"
        )
    return values


def available_properties(section: Any, *names: str) -> dict[str, float]:
    """Read whatever of ``names`` is present, skipping the rest. Never raises."""
    values: dict[str, float] = {}
    for name in names:
        try:
            values[name] = float(getattr(section, name))
        except (AttributeError, TypeError, ValueError):
            continue
    return values
