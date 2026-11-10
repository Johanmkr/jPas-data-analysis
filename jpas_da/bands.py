"""Canonical J-PAS filter system as used by the miniJPAS export.

Every module that touches the 60-band axis imports its band order, names and
wavelengths from here, so the order cannot drift between the raw-to-interim
step, the feature builder, the models and the notebook.

The order is deliberately *not* strictly wavelength-sorted. It keeps the 54
narrow bands contiguous (they form the evenly sampled J-spectrum, which is the
physically meaningful axis) and appends the remaining filters as their own
blocks:

    block      n   members
    narrow    54  J0378 ... J0910, FWHM ~145 A, 100 A grid (120 A for the first step)
    medium     1  uJAVA  (FWHM 495 A, centred 3497 A)
    broad      4  uJPAS, gSDSS, rSDSS, iSDSS
    highpass   1  J1007  (high-pass, centred 9316 A)

Wavelength units: ``wavelength_aa`` is **always Angstrom**, for all 60 bands.
The narrow filter names encode nanometres (``J0378`` = 378 nm = 3780 A), so the
name is multiplied by 10 when the table is built; everything else is already in
Angstrom.

Filter taxonomy, the 145 A / 100 A grid and the uJAVA/J1007 central
wavelengths are from the miniJPAS description (Bonoli et al. 2021; see also
arXiv:2207.10101, Sec. 2.1). ``wavelength_source`` records where each value
came from:

    name     the value is encoded in the filter name (narrow bands, exact)
    paper    taken from the miniJPAS filter description
    nominal  a nominal SDSS AB effective wavelength, kept only so the band can
             be ordered and plotted; no analysis depends on its exact value
"""

from __future__ import annotations

from dataclasses import dataclass

NARROW = tuple(["J0378"] + [f"J{w:04d}" for w in range(390, 911, 10)])
"""The 54 narrow bands, in wavelength order (3780-9100 A)."""

MEDIUM = ("uJAVA",)
BROAD = ("uJPAS", "gSDSS", "rSDSS", "iSDSS")
HIGHPASS = ("J1007",)

BLOCK_ORDER = ("narrow", "medium", "broad", "highpass")

# Nominal effective wavelengths for the filters whose names do not encode one.
_NOMINAL_WAVELENGTH = {
    "uJAVA": 3497.0,  # medium band, Bonoli et al. 2021
    "uJPAS": 3551.0,  # nominal SDSS u
    "gSDSS": 4686.0,  # nominal SDSS g
    "rSDSS": 6165.0,  # nominal SDSS r
    "iSDSS": 7481.0,  # nominal SDSS i
    "J1007": 9316.0,  # high-pass, Bonoli et al. 2021
}

_BLOCK_SOURCE = {
    "narrow": "name",
    "medium": "paper",
    "broad": "nominal",
    "highpass": "paper",
}

_FILTER_WIDTH_AA = {"narrow": 145.0, "medium": 495.0, "broad": None, "highpass": None}


@dataclass(frozen=True)
class Band:
    """One photometric band."""

    name: str
    block: str
    wavelength_aa: float
    width_aa: float | None
    wavelength_source: str


def _make_band(name: str, block: str) -> Band:
    if block == "narrow":
        # Filter names encode nanometres: J0378 -> 378 nm -> 3780 A.
        wavelength = float(name[1:]) * 10.0
        source = "name"
    else:
        wavelength = _NOMINAL_WAVELENGTH[name]
        source = _BLOCK_SOURCE[block]
    return Band(name, block, wavelength, _FILTER_WIDTH_AA[block], source)


BANDS = tuple(
    _make_band(name, block)
    for block in BLOCK_ORDER
    for name in {"narrow": NARROW, "medium": MEDIUM, "broad": BROAD, "highpass": HIGHPASS}[block]
)
"""All 60 bands, in canonical order."""

BAND_NAMES = tuple(b.name for b in BANDS)
"""Canonical band order as plain names."""

N_BANDS = len(BANDS)

BAND_INDEX = {name: i for i, name in enumerate(BAND_NAMES)}
"""Position of each band in the canonical axis."""

# The raw export names the relative-error column of each band with an ``e``
# prefix, e.g. ``J0450`` -> ``eJ0450``, ``uJAVA`` -> ``euJAVA``.
ERROR_COLUMN = {name: f"e{name}" for name in BAND_NAMES}


def select(block: str) -> tuple[str, ...]:
    """Band names belonging to one block."""
    if block not in BLOCK_ORDER:
        raise ValueError(f"unknown block {block!r}; expected one of {BLOCK_ORDER}")
    return tuple(b.name for b in BANDS if b.block == block)


def wavelengths(block: str | None = None) -> list[float]:
    """Wavelengths in Angstrom, in canonical band order."""
    return [b.wavelength_aa for b in BANDS if block is None or b.block == block]
