"""Unsupervised evaluation shared by the VAE and the baselines.

Every model is scored in the same space with the same noise model, so the
numbers are comparable:

* **Reconstruction**: per object ``chi2 = sum_j m_j (g_j - g_hat_j)^2 / w_j^2``
  and ``chi2/dof`` with ``dof = sum_j m_j`` (the number of measured bands; the
  model's own degrees of freedom are not subtracted, so a perfect noise model
  gives ~1 only for a model with few parameters per object). Reported overall
  and binned by brightness S/N (``1/sig_logr``), tile and quality class.
* **Latent use** (VAE only): KL per latent dimension, averaged over objects,
  and the number of *active* dimensions (mean KL above :data:`ACTIVE_KL`
  nats). A dimension with KL ~ 0 carries no information about the input.

Labels are never read here; label-based probes belong to a later goal.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np

SNR_EDGES = (0.0, 3.0, 5.0, 10.0, 30.0, np.inf)
"""Bins of the brightness S/N ``1/sig_logr`` (median 6.4, 5th percentile 2.5)."""

ACTIVE_KL = 0.01
"""Mean KL (nats) above which a latent dimension counts as active."""


def chi2_per_object(
    g: np.ndarray, g_hat: np.ndarray, w2: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(chi2, dof)`` per object."""
    mask = np.asarray(mask, dtype=np.float64)
    residual = (np.asarray(g, np.float64) - np.asarray(g_hat, np.float64)) ** 2
    chi2 = (residual / np.maximum(np.asarray(w2, np.float64), 1e-12) * mask).sum(axis=1)
    return chi2, mask.sum(axis=1)


def _summary(chi2: np.ndarray, dof: np.ndarray) -> dict[str, float]:
    if len(chi2) == 0:
        return {"n": 0}
    reduced = chi2 / np.maximum(dof, 1.0)
    return {
        "n": len(chi2),
        "chi2_dof_pooled": float(chi2.sum() / dof.sum()),
        "chi2_dof_median": float(np.median(reduced)),
        "chi2_dof_p90": float(np.percentile(reduced, 90)),
    }


def snr_label(lo: float, hi: float) -> str:
    return f"{lo:g}-{hi:g}" if np.isfinite(hi) else f">{lo:g}"


def reconstruction_report(
    chi2: np.ndarray,
    dof: np.ndarray,
    sig_logr: np.ndarray,
    tile_id: np.ndarray,
    quality: np.ndarray,
) -> dict[str, object]:
    """Pooled and median chi2/dof overall and by S/N bin, tile and quality."""
    snr = 1.0 / np.maximum(np.asarray(sig_logr, np.float64), 1e-12)
    report: dict[str, object] = {"overall": _summary(chi2, dof)}
    report["by_snr"] = {
        snr_label(lo, hi): _summary(chi2[(snr >= lo) & (snr < hi)], dof[(snr >= lo) & (snr < hi)])
        for lo, hi in pairwise(SNR_EDGES)
    }
    report["by_tile"] = {
        str(int(t)): _summary(chi2[tile_id == t], dof[tile_id == t]) for t in np.unique(tile_id)
    }
    report["by_quality"] = {
        str(q): _summary(chi2[quality == q], dof[quality == q]) for q in np.unique(quality)
    }
    return report


def latent_report(kl: np.ndarray) -> dict[str, object]:
    """Mean KL per dimension (sorted descending) and the active-dimension count."""
    mean_kl = np.asarray(kl, np.float64).mean(axis=0)
    return {
        "kl_per_dim": [float(x) for x in mean_kl],
        "kl_total": float(mean_kl.sum()),
        "active_dims": int((mean_kl > ACTIVE_KL).sum()),
    }
