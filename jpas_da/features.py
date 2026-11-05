"""Interim object table -> model-ready representation.

The model input is the noise-scaled inverse hyperbolic sine of the flux::

    x_b      = asinh(f_b / b_{p(b)}) * s_{p(b)}
    sigma_x  = sigma_f / sqrt(f_b**2 + b_b**2) * s_{p(b)}
    f_b      = b_b * sinh(x_b / s_{p(b)})

with ``b_b`` the per-band soft scale from :mod:`jpas_da.dataset` and ``s_p``
a per-band *presentation* scale chosen so all 60 channels have comparable
spread.

Why asinh and not raw flux, ``log``, or ``f/sigma`` -- all measured on the
shipped sample (3.86M pixels):

* Asinh is linear when ``|f| < b`` and logarithmic when ``|f| >> b``. 41% of
  pixels have |SNR| < 1 and 21.7% have negative flux, so a plain ``log`` is
  undefined on a fifth of the data and would need an arbitrary floor. Asinh is
  defined everywhere, invertible, and needs no imputation.
* The output noise stays nearly flat over the flux range. In flux units the
  propagated sigma spans 4.8x across |f| deciles; with ``b_b = median sigma``
  it spans 0.53..0.43 for the nine faintest deciles (dropping only on the
  brightest, where the measurement really is better): corr(sigma_x, |f|) =
  -0.01. A heteroscedastic Gaussian likelihood wants exactly that.
* The geometry is sane. In raw flux, the brightest 0.1% of pixels carry 99.9%
  of sum(f^2), so L2 distance is a fit to a few thousand bright pixels; only
  60% of the 50 nearest neighbours of a sample object survive the switch to
  asinh. ``f/sigma`` is worse still: it inverts the problem (sigma_x spans 65x
  across flux deciles), turns the geometry into an SNR geometry, and is
  undefined on the 0.94% of pixels with ``relerr == 0``.
* Asinh space is close to Gaussian (skew 1.35, kurtosis 4.6) and its first PC
  is per-object brightness at 69.5% of the variance, with 65% of the total coming
  from the mean level and 35% from SED shape. Both are kept: brightness is a
  real physical axis (distance/luminosity), so it stays in the channels rather
  than being normalised away.

Masked pixels (``mask == 0``, sigma undefined) get ``x = 0``, the asinh origin,
i.e. "no flux measurement". The mask channel tells the model and the loss to
ignore them; ``x = 0`` is never mistaken for a detection because every real
non-detection has ``|x|`` of order the noise, not zero.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from jpas_da.bands import BAND_NAMES
from jpas_da.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR

app = typer.Typer()

OBJECTS_FILE = INTERIM_DATA_DIR / "objects.parquet"
BAND_SCALES_FILE = INTERIM_DATA_DIR / "band_scales.parquet"
REPRESENTATION_FILE = PROCESSED_DATA_DIR / "jpas_asinh.parquet"

# Presentation scaling: one global factor for all bands. It only fixes the
# numerical magnitude of the channels (asinh(x)/s is monotonic, so it cannot
# change a distance ordering) and is kept in the file so inversion is exact.
PRESENTATION_SCALE = 2.0


def asinh_transform(flux: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``asinh(f / b)``; masked (NaN) flux maps to 0."""
    out = np.arcsinh(np.divide(flux, scale, out=np.zeros_like(flux), where=np.isfinite(flux)))
    out[~np.isfinite(flux)] = 0.0
    return out


def asinh_error(flux: np.ndarray, sigma: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Propagate ``sigma_f`` through ``asinh(f/b)``: ``d asinh/d f = 1/sqrt(f^2+b^2)``."""
    denominator = np.sqrt(flux**2 + scale**2)
    return np.where(np.isfinite(sigma), sigma / denominator, np.nan)


def asinh_inverse(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`asinh_transform`."""
    return scale * np.sinh(x)


def build_representation(
    objects: pd.DataFrame, scales: pd.DataFrame, presentation: float = PRESENTATION_SCALE
) -> pd.DataFrame:
    """Build ``x``, ``sigma_x`` and ``mask`` from the interim object table."""
    if list(scales["band"]) != list(BAND_NAMES):
        raise ValueError("band scales are not in canonical band order")
    scale = scales["b_j"].to_numpy(dtype=float)

    flux = objects[[f"flux_{b}" for b in BAND_NAMES]].to_numpy(dtype=float)
    sigma = objects[[f"sigma_{b}" for b in BAND_NAMES]].to_numpy(dtype=float)
    mask = objects[[f"mask_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.int8)

    x = asinh_transform(flux, scale) * presentation
    sigma_x = asinh_error(flux, sigma, scale) * presentation
    # A masked pixel has a flux but no measured uncertainty. Put it at the
    # asinh origin, which is unambiguous because real non-detections scatter
    # around |x| ~ 1, not 0.
    x[mask == 0] = 0.0
    sigma_x[mask == 0] = np.nan

    if not np.isfinite(x).all():
        raise ValueError("non-finite values in the asinh representation")
    if np.any(np.isfinite(sigma_x) & (sigma_x <= 0)):
        raise ValueError("non-positive propagated uncertainty")
    if np.any(np.isfinite(sigma_x) & (mask == 0)):
        raise ValueError("sigma defined where the mask says it is not")

    def frame(values: np.ndarray, prefix: str, dtype=None) -> pd.DataFrame:
        return pd.DataFrame(values, columns=[f"{prefix}{b}" for b in BAND_NAMES], dtype=dtype)

    identifiers = objects[["object_id", "tile_id"]].reset_index(drop=True)
    representation = pd.concat(
        [identifiers, frame(x, "x_"), frame(sigma_x, "sigma_x_"), frame(mask, "mask_")], axis=1
    )

    logger.info(
        f"x: spread {np.nanstd(x):.2f}, |x| p99 {np.nanpercentile(np.abs(x), 99):.2f}; "
        f"sigma_x: median {np.nanmedian(sigma_x):.3f}, p99 {np.nanpercentile(sigma_x, 99):.2f}"
    )
    logger.info(
        f"masked pixels: {int((mask == 0).sum()):,} "
        f"({100 * (mask == 0).mean():.2f}%); "
        f"objects with a full 60-band spectrum: {int((mask == 1).all(axis=1).sum()):,}"
    )
    return representation


def check_roundtrip(
    representation: pd.DataFrame, objects: pd.DataFrame, scales: pd.DataFrame
) -> None:
    """Assert ``x -> f`` recovers the flux of every defined pixel."""
    scale = scales["b_j"].to_numpy(dtype=float)
    x = representation[[f"x_{b}" for b in BAND_NAMES]].to_numpy(dtype=float) / PRESENTATION_SCALE
    recovered = asinh_inverse(x, scale)
    flux = objects[[f"flux_{b}" for b in BAND_NAMES]].to_numpy(dtype=float)
    mask = objects[[f"mask_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.int8)
    residual = np.abs(recovered - flux)[mask == 1]
    tolerance = 1e-8 * (1.0 + np.abs(flux[mask == 1]))
    if not np.all(residual <= tolerance):
        raise ValueError(f"asinh round-trip failed; worst residual {residual.max():.3e}")


def build(
    objects_path: Path = OBJECTS_FILE,
    scales_path: Path = BAND_SCALES_FILE,
    output_path: Path = REPRESENTATION_FILE,
) -> pd.DataFrame:
    objects = pd.read_parquet(objects_path)
    scales = pd.read_parquet(scales_path)
    logger.info(f"Loaded {len(objects):,} objects and {len(scales)} band scales")

    representation = build_representation(objects, scales)
    check_roundtrip(representation, objects, scales)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    representation.to_parquet(output_path, index=False)
    logger.success(f"Wrote {output_path} ({len(representation):,} x {representation.shape[1]})")
    return representation


def load_arrays(
    path: Path = REPRESENTATION_FILE,
    folds_path: Path | None = None,
) -> dict[str, np.ndarray]:
    """Load the representation as plain arrays for model training.

    Returns ``x``, ``sigma_x`` and ``mask`` of shape ``(n_objects, n_bands)``
    in canonical band order, plus ``object_id``, ``tile_id`` and, when a fold
    assignment is available, ``fold``. ``sigma_x`` is NaN where ``mask == 0``.
    """
    representation = pd.read_parquet(path)
    arrays = {
        "x": representation[[f"x_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float32),
        "sigma_x": representation[[f"sigma_x_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.float32),
        "mask": representation[[f"mask_{b}" for b in BAND_NAMES]].to_numpy(dtype=np.int8),
        "object_id": representation["object_id"].to_numpy(),
        "tile_id": representation["tile_id"].to_numpy(dtype=np.int64),
    }
    if folds_path is None:
        folds_path = INTERIM_DATA_DIR / "folds.parquet"
    if folds_path.exists():
        folds = pd.read_parquet(folds_path)
        aligned = representation[["object_id"]].merge(folds, on="object_id", how="left")
        if aligned["fold"].isna().any():
            raise ValueError("fold assignment does not cover every object")
        arrays["fold"] = aligned["fold"].to_numpy(dtype=np.int64)
    return arrays


@app.command()
def main(
    objects_path: Path = OBJECTS_FILE,
    scales_path: Path = BAND_SCALES_FILE,
    output_path: Path = REPRESENTATION_FILE,
):
    build(objects_path, scales_path, output_path)


if __name__ == "__main__":
    app()
