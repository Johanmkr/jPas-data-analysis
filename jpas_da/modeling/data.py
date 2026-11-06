"""Load the model-ready arrays and cut them into the roles of one rotation.

The arrays are written by :mod:`jpas_da.features` (``make features``); the roles
come from :func:`jpas_da.splits.assign_roles`. Training, prediction and the
baselines all go through :func:`load`, :func:`split_indices` and
:func:`model_inputs`, so they cannot disagree about which rows are held out or
how the encoder input is laid out.

Encoder input layout, shared by every encoder:

* ``narrow``: ``(N, C, 54)`` -- the configured channels over the 54 narrow bands,
  which sit on the evenly spaced 100 A grid;
* ``other``: ``(N, C * 6)`` -- the same channels for uJAVA, uJPAS, gSDSS, rSDSS,
  iSDSS and J1007, flattened band-major;
* ``cond``: ``(N, K)`` -- the conditioning variables (``logr`` and optionally
  ``sig_logr``), given to encoder *and* decoder.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from jpas_da.bands import BAND_INDEX, select
from jpas_da.features import MODEL_DATA_FILE
from jpas_da.splits import ROLES, assign_roles

NARROW_INDEX = np.array([BAND_INDEX[b] for b in select("narrow")])
OTHER_INDEX = np.array(
    [BAND_INDEX[b] for block in ("medium", "broad", "highpass") for b in select(block)]
)

CHANNELS = ("g", "v", "mask")
"""Per-band encoder channels."""

CONDITIONS = ("logr", "sig_logr")
"""Per-object conditioning variables."""

REQUIRED = {"g", "v", "mask", "logr", "sig_logr", "object_id", "tile_id", "holdout", "quality"}


def load(path: Path = MODEL_DATA_FILE) -> dict[str, np.ndarray]:
    """Load the arrays written by :func:`jpas_da.features.build`."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make data` first")
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    missing = REQUIRED - arrays.keys()
    if missing:
        raise ValueError(f"{path} is missing keys {sorted(missing)}; rerun `make data`")
    return arrays


def roles(arrays: dict[str, np.ndarray], rotation: int) -> np.ndarray:
    """Role of every object in ``rotation`` (see :mod:`jpas_da.splits`)."""
    return assign_roles(arrays["tile_id"], arrays["holdout"], arrays["quality"], rotation)


def split_indices(arrays: dict[str, np.ndarray], rotation: int) -> dict[str, np.ndarray]:
    """Row indices of ``train``, ``val``, ``test`` and ``holdout`` for one rotation."""
    assigned = roles(arrays, rotation)
    return {role: np.flatnonzero(assigned == role) for role in (*ROLES, "holdout")}


def model_inputs(
    arrays: dict[str, np.ndarray],
    channels: tuple[str, ...] = CHANNELS,
    conditions: tuple[str, ...] = ("logr",),
) -> dict[str, np.ndarray]:
    """Build the encoder input blocks, float32, for every object."""
    unknown = set(channels) - set(CHANNELS)
    if unknown or not channels:
        raise ValueError(f"channels must be a non-empty subset of {CHANNELS}, got {channels}")
    unknown = set(conditions) - set(CONDITIONS)
    if unknown or "logr" not in conditions:
        raise ValueError(f"conditions must include 'logr' and be a subset of {CONDITIONS}")
    stacked = np.stack([arrays[name].astype(np.float32) for name in channels], axis=1)
    other = stacked[:, :, OTHER_INDEX].transpose(0, 2, 1).reshape(len(stacked), -1)
    return {
        "narrow": np.ascontiguousarray(stacked[:, :, NARROW_INDEX]),
        "other": np.ascontiguousarray(other),
        "cond": np.stack([arrays[name].astype(np.float32) for name in conditions], axis=1),
    }
