"""The asinh representation, kept as the input of the PCA baseline.

The VAE does not use it (see :mod:`jpas_da.features` for the model inputs). It
stays because a model *without* a noise likelihood -- PCA, kNN, a plain-MSE
autoencoder -- needs a space where the noise is roughly uniform, and asinh gives
exactly that::

    x_j       = asinh(f_j / b_j)
    sigma_x_j = sigma_j / sqrt(f_j**2 + b_j**2)
    f_j       = b_j * sinh(x_j)

with ``b_j`` the per-band soft scale, the median error of band ``j``. Asinh is
linear for ``|f| < b`` and logarithmic for ``|f| >> b``, defined for negative
flux, and exactly invertible. On this export the propagated noise is close to
flat across the flux range (corr(sigma_x, |f|) = -0.01), where in flux units it
spans 4.8x across |f| deciles.

``b_j`` is a sample statistic, so it is fitted on the training tiles of a
rotation only (:func:`fit_band_scales`), never on the whole catalogue.
"""

from __future__ import annotations

import numpy as np


def fit_band_scales(sigma: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-band soft scale ``b_j = median sigma_j`` over the measured pixels given.

    Pass only the training rows of a rotation.
    """
    sigma = np.asarray(sigma, dtype=np.float64)
    measured = (np.asarray(mask) > 0) & np.isfinite(sigma) & (sigma > 0)
    scales = np.array(
        [
            np.median(sigma[measured[:, j], j]) if measured[:, j].any() else np.nan
            for j in range(sigma.shape[1])
        ]
    )
    if not np.isfinite(scales).all() or (scales <= 0).any():
        bad = np.flatnonzero(~np.isfinite(scales) | (scales <= 0)).tolist()
        raise ValueError(f"could not derive a positive soft scale for bands {bad}")
    return scales


def transform(flux: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """``asinh(f / b)``; non-finite flux maps to 0."""
    flux = np.asarray(flux, dtype=np.float64)
    out = np.arcsinh(np.divide(flux, scale, out=np.zeros_like(flux), where=np.isfinite(flux)))
    out[~np.isfinite(flux)] = 0.0
    return out


def error(flux: np.ndarray, sigma: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Propagate ``sigma_f`` through ``asinh(f/b)``: ``d asinh/d f = 1/sqrt(f^2+b^2)``."""
    denominator = np.sqrt(np.asarray(flux, dtype=np.float64) ** 2 + scale**2)
    return np.where(np.isfinite(sigma), sigma / denominator, np.nan)


def inverse(x: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Exact inverse of :func:`transform`."""
    return scale * np.sinh(x)
