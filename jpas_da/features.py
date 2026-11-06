"""Interim object table -> model-ready arrays: broad-band RMS normalisation.

Every band of an object is divided by one number, its broad-band brightness::

    r_bb     = sqrt( mean over measured k in {gSDSS, rSDSS, iSDSS} of f_k^2 )
    g_j      = f_j / r_bb                                   all 60 bands
    v_j      = sqrt( sigma_j^2 + (0.026 f_j)^2 ) / r_bb     0 where mask_j = 0
    logr     = log r_bb
    sig_logr = sqrt( sum_k m_k f_k^2 sigma_k^2 ) / (n r_bb^2)   delta method

Why this normaliser (all measured on the shipped export):

* **Broad bands, not narrow ones.** Narrow bands have median S/N ~1.1; over the
  54 of them, noise makes up a median 69% of the RMS (``sqrt(sum sigma^2)/r``),
  so a narrow-band brightness is mostly noise and biased high. ``log r_bb``
  has median S/N 6.4 (``1/sig_logr``; 5th percentile 2.5).
* **One scale for all 60 bands.** Narrow and broad bands stay on a common
  footing, so narrow-versus-broad colours survive. There is no projection onto
  the unit sphere.
* **No sample statistic.** Each object is normalised by itself, so nothing
  leaks across tiles.
* **The noise of r_bb is shared by every band.** Dividing by a noisy scalar
  adds a perfectly correlated error ``g_j * sig_logr``; the loss can add it as a
  diagonal term (``loss.common_mode``), which overstates its independence.
* ``0.026`` is the miniJPAS relative calibration scatter (~0.028 mag), a survey
  constant, not fitted here.

A masked band (no usable error) keeps ``g = 0`` and ``v = 0``, so the loss gives
it exactly zero weight. A masked norm band is left out of ``r_bb``; no object
has all three masked.

``r_bb`` is still noisy for the faintest objects and is not debiased here.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from jpas_da.bands import BAND_INDEX, BAND_NAMES, N_BANDS
from jpas_da.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

OBJECTS_FILE = INTERIM_DATA_DIR / "objects.parquet"
MODEL_DATA_FILE = PROCESSED_DATA_DIR / "vae_inputs.npz"

NORM_BANDS = ("gSDSS", "rSDSS", "iSDSS")
"""Bands whose RMS is the per-object brightness."""

RELATIVE_CALIB_FLOOR = 0.026
"""miniJPAS relative calibration scatter as a fraction of flux (~0.028 mag)."""


def normalise(
    flux: np.ndarray,
    sigma: np.ndarray,
    mask: np.ndarray,
    floor_frac: float = RELATIVE_CALIB_FLOOR,
    norm_index: list[int] | None = None,
) -> dict[str, np.ndarray]:
    """Divide every band by the RMS of the norm bands.

    ``flux`` and ``sigma`` are ``(n_objects, n_bands)``; ``sigma`` may be NaN
    where ``mask`` is 0. ``norm_index`` selects the norm bands (default: g, r, i
    in canonical order).

    Returns ``g``, ``v`` (float32), ``mask`` (int8), ``logr`` and ``sig_logr``
    (float32, shape ``(n_objects,)``).
    """
    flux = np.asarray(flux, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    mask = np.asarray(mask, dtype=np.float64)
    if flux.shape != sigma.shape or flux.shape != mask.shape:
        raise ValueError(f"shape mismatch: flux {flux.shape}, sigma {sigma.shape}")
    if not np.isfinite(flux).all():
        raise ValueError("non-finite flux in the interim table")
    if norm_index is None:
        norm_index = [BAND_INDEX[b] for b in NORM_BANDS]

    total_sigma = np.sqrt(np.where(mask > 0, sigma, 0.0) ** 2 + (floor_frac * flux) ** 2)

    f_norm = flux[:, norm_index]
    m_norm = mask[:, norm_index]
    s_norm = total_sigma[:, norm_index]
    n_norm = m_norm.sum(axis=1)
    if np.any(n_norm == 0):
        raise ValueError(f"{int((n_norm == 0).sum())} objects have every norm band masked")
    r2 = (f_norm**2 * m_norm).sum(axis=1) / n_norm
    if np.any(r2 <= 0):
        raise ValueError(f"{int((r2 <= 0).sum())} objects have zero norm-band brightness")
    r = np.sqrt(r2)

    g = np.where(mask > 0, flux / r[:, None], 0.0)
    v = np.where(mask > 0, total_sigma / r[:, None], 0.0)
    logr = np.log(r)
    sig_logr = np.sqrt((m_norm * f_norm**2 * s_norm**2).sum(axis=1)) / (n_norm * r2)

    if not (np.isfinite(g).all() and np.isfinite(v).all()):
        raise ValueError("non-finite values after normalisation")
    if not (np.isfinite(logr).all() and np.isfinite(sig_logr).all()):
        raise ValueError("non-finite brightness or brightness error")

    return {
        "g": g.astype(np.float32),
        "v": v.astype(np.float32),
        "mask": mask.astype(np.int8),
        "logr": logr.astype(np.float32),
        "sig_logr": sig_logr.astype(np.float32),
    }


def build(
    objects_path: Path = OBJECTS_FILE, output_path: Path = MODEL_DATA_FILE
) -> dict[str, np.ndarray]:
    """Write the model-ready arrays with the identifiers the loader needs."""
    objects = pd.read_parquet(objects_path)
    if objects["object_id"].duplicated().any():
        raise ValueError("duplicated object_id in the interim table")

    flux = objects[[f"flux_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    sigma = objects[[f"sigma_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float64)
    mask = objects[[f"mask_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.int8)
    arrays = normalise(flux, sigma, mask)

    arrays["object_id"] = objects["object_id"].to_numpy(dtype=str)
    arrays["tile_id"] = objects["tile_id"].to_numpy(dtype=np.int64)
    arrays["holdout"] = objects["holdout"].to_numpy(dtype=bool)
    arrays["quality"] = objects["quality"].to_numpy(dtype=str)
    arrays["n_bands"] = np.array(N_BANDS, dtype=np.int64)
    arrays["relative_calib_floor"] = np.array(RELATIVE_CALIB_FLOOR, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    snr = 1.0 / np.maximum(arrays["sig_logr"], 1e-12)
    logger.success(
        f"Wrote {output_path}: {len(arrays['g']):,} objects, {N_BANDS} bands; "
        f"r_bb S/N median {np.median(snr):.2f}, p5 {np.percentile(snr, 5):.2f}; "
        f"masked {100 * (1 - arrays['mask'].mean()):.2f}%"
    )
    return arrays


@app.command()
def main(objects_path: Path = OBJECTS_FILE, output_path: Path = MODEL_DATA_FILE):
    build(objects_path, output_path)


if __name__ == "__main__":
    app()
