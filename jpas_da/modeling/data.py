"""Model-ready tensors for the VAE: RMS shape/scale decomposition.

The interim table stores flux ``f``, absolute error ``sigma`` and a validity
mask per band. This module turns one object's row into

* ``g = f / r`` with ``r = sqrt(sum_j f_j^2 mask_j)`` -- the SED shape,
  unit-norm over the measured bands;
* ``v = sqrt(sigma^2 + (floor_frac * f)^2) / r`` -- the propagated shape
  error, carrying the multiplicative relative-calibration term. Masked bands
  get ``v = 0`` so the loss can give them exactly zero weight;
* ``logr`` and ``sig_logr`` -- brightness and its statistical uncertainty.

Dividing flux and error by the same per-object scalar leaves every Gaussian
likelihood term invariant, so this is a conditioning choice, not a model of the
data. It is computed from each object alone: no sample statistic enters, so it
cannot leak across the leave-one-tile-out folds.

The additive per-band floor ``s_j`` is **not** stored here. It has to be fitted
on the training fold only, so :mod:`jpas_da.modeling.train` derives it from the
rows it trains on.

Run with ``python -m jpas_da.modeling.data`` (``make model-data``).
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from jpas_da.bands import BAND_NAMES, N_BANDS
from jpas_da.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

OBJECTS_FILE = INTERIM_DATA_DIR / "objects.parquet"
FOLDS_FILE = INTERIM_DATA_DIR / "folds.parquet"
MODEL_DATA_FILE = PROCESSED_DATA_DIR / "vae_inputs.npz"

# Relative photometric-calibration scatter of miniJPAS, 0.028 mag as reported
# for the narrow-band comparison. In fractional flux this is ~0.026. It is a
# survey constant, not a fitted quantity.
RELATIVE_CALIB_FLOOR = 0.026


def normalise(
    flux: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    floor_frac: float = RELATIVE_CALIB_FLOOR,
) -> dict[str, np.ndarray]:
    """Per-object RMS normalisation of flux, error and mask.

    ``flux`` and ``sigma`` are ``(n_objects, n_bands)`` in canonical band order,
    ``mask`` is 0 where the archive quoted no error. A masked band still has a
    flux but no usable uncertainty, so it is excluded from the norm and given
    zero shape error.

    Returns ``g``, ``v``, ``mask`` (float32, mask as int8), ``logr`` and
    ``sig_logr`` (float32, shape ``(n_objects,)``).
    """
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if flux.shape != sigma.shape or flux.shape != mask.shape:
        raise ValueError(f"shape mismatch: flux {flux.shape}, sigma {sigma.shape}")
    if not np.isfinite(flux).all():
        raise ValueError("non-finite flux in the interim table")

    r = np.sqrt((flux**2 * mask).sum(axis=1))
    if np.any(r <= 0):
        raise ValueError(f"{int((r <= 0).sum())} objects have zero masked norm")

    # The archive quotes a signed relative error, so the absolute error is
    # flux * relerr; here sigma is already absolute and non-negative.
    sigma = np.sqrt(sigma**2 + (floor_frac * flux) ** 2)

    g = flux / r[:, None]
    v = sigma / r[:, None]
    v = np.where(mask > 0, v, 0.0)
    g = np.where(mask > 0, g, 0.0)

    logr = np.log(r)
    sig_logr = np.sqrt((mask * g**2 * v**2).sum(axis=1))

    if not np.isfinite(g).all() or not np.isfinite(v).all():
        raise ValueError("non-finite values after normalisation")
    if not np.isfinite(logr).all() or not np.isfinite(sig_logr).all():
        raise ValueError("non-finite brightness or brightness error")

    return {
        "g": g.astype(np.float32),
        "v": v.astype(np.float32),
        "mask": mask.astype(np.int8),
        "logr": logr.astype(np.float32),
        "sig_logr": sig_logr.astype(np.float32),
    }


def build(
    objects_path: Path = OBJECTS_FILE,
    folds_path: Path = FOLDS_FILE,
    output_path: Path = MODEL_DATA_FILE,
) -> dict[str, np.ndarray]:
    """Write the model-ready arrays, aligned with the fold assignment."""
    objects = pd.read_parquet(objects_path)
    if objects[["object_id", "tile_id"]].duplicated().any():
        raise ValueError("duplicated object_id/tile_id in the interim table")

    flux = objects[[f"flux_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    sigma = objects[[f"sigma_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    mask = objects[[f"mask_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.int8)
    arrays = normalise(flux, sigma, mask)

    if folds_path.exists():
        folds = pd.read_parquet(folds_path)
        aligned = objects[["object_id"]].merge(folds, on="object_id", how="left")
        if aligned["fold"].isna().any():
            raise ValueError("fold assignment does not cover every object")
        arrays["fold"] = aligned["fold"].to_numpy(dtype=np.int64)
        logger.info(f"aligned fold assignment for {len(aligned):,} objects")
    else:
        logger.warning(f"{folds_path} not found; writing arrays without folds")

    arrays["object_id"] = objects["object_id"].to_numpy(dtype=str)
    arrays["tile_id"] = objects["tile_id"].to_numpy(dtype=np.int64)
    arrays["n_bands"] = np.array(N_BANDS, dtype=np.int64)
    arrays["relative_calib_floor"] = np.array(RELATIVE_CALIB_FLOOR, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    logger.success(
        f"Wrote {output_path}: {len(arrays['g']):,} objects, {N_BANDS} bands, "
        f"median |g| {np.abs(arrays['g']).mean():.3f}, "
        f"masked {100 * (1 - arrays['mask'].mean()):.2f}%"
    )
    return arrays


def load(path: Path = MODEL_DATA_FILE) -> dict[str, np.ndarray]:
    """Load the model-ready arrays written by :func:`build`."""
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found; run `make model-data` first")
    with np.load(path, allow_pickle=False) as handle:
        arrays = {key: handle[key] for key in handle.files}
    required = {"g", "v", "mask", "logr", "sig_logr", "fold", "object_id"}
    missing = required - arrays.keys()
    if missing:
        raise ValueError(f"{path} is missing keys: {sorted(missing)}")
    return arrays


@app.command()
def main(
    objects_path: Path = OBJECTS_FILE,
    folds_path: Path = FOLDS_FILE,
    output_path: Path = MODEL_DATA_FILE,
):
    build(objects_path, folds_path, output_path)


if __name__ == "__main__":
    app()
