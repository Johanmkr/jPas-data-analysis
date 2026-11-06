"""Noise model and reconstruction terms.

The per-cell variance is known, not learned::

    w_j^2 = v_j^2 + eps_j^2 (+ (g_j * sig_logr)^2)      0 where mask_j = 0

* ``v_j`` is the propagated measurement error, already carrying the 0.026
  relative calibration term (:mod:`jpas_da.features`).
* ``eps`` is a fixed floor *relative to the object's brightness*: no band is
  known better than ``eps * r_bb``. It is needed because, after dividing by
  ``r_bb``, very bright objects really are measured very precisely -- the
  smallest ``v`` in the catalogue belongs to an object ~8000x brighter than the
  median -- and without a floor a handful of them hold most of the Fisher
  information. It is not that the archive error vanishes at low flux: it does
  not (no measured pixel has sigma below 0.064x its band median). A cap
  proportional to ``|g|`` would not help: the 0.026 term already caps per-band
  S/N at ~38, and the extreme weights sit on near-zero-flux bands of bright
  objects, where ``|g| ~ 0``.
* ``(g_j * sig_logr)^2`` is the error that dividing by a noisy ``r_bb`` adds to
  every band. It is perfectly correlated across bands; the diagonal term
  overstates its independence (``loss.common_mode``).

A reconstruction term scores ``mu`` against ``g`` given ``w2`` and the mask and
returns one value per object, summed over bands::

    loss(g, mu, w2, mask) -> Tensor   # shape (N,)

The KL term belongs to the training loop.
"""

from __future__ import annotations

import math
from typing import Protocol

import torch


class ReconstructionLoss(Protocol):
    def __call__(
        self,
        g: torch.Tensor,
        mu: torch.Tensor,
        w2: torch.Tensor,
        mask: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor: ...


def noise_variance(
    g: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    sig_logr: torch.Tensor,
    eps: torch.Tensor | float,
    common_mode: bool = True,
) -> torch.Tensor:
    """Per-cell variance ``w^2``, zero on masked bands."""
    eps = torch.as_tensor(eps, dtype=v.dtype, device=v.device)
    w2 = v**2 + eps**2
    if common_mode:
        w2 = w2 + (g * sig_logr.unsqueeze(-1)) ** 2
    return w2 * mask


def gaussian_nll(g, mu, w2, mask, **_: object) -> torch.Tensor:
    """Gaussian negative log-likelihood with known per-cell variance.

    ``w2`` does not depend on the parameters, so the ``log(2 pi w2)`` term is a
    constant: it changes the reported value (making it a proper NLL comparable
    across models), not the gradients.
    """
    safe = w2.clamp_min(1e-12)
    cell = 0.5 * ((g - mu) ** 2 / safe + torch.log(2 * math.pi * safe))
    return (cell * mask).sum(-1)


def student_t(g, mu, w2, mask, *, dof: float = 5.0, **_: object) -> torch.Tensor:
    """Heavy-tailed alternative with the same scale ``w``, normalisation dropped.

    Use it to test whether the latent geometry is set by gross outliers: if the
    embedding is unchanged, the Gaussian was not outlier-driven.
    """
    if dof <= 0:
        raise ValueError("student_t dof must be positive")
    residual = (g - mu) ** 2 / w2.clamp_min(1e-12)
    cell = 0.5 * (dof + 1.0) * torch.log1p(residual / dof)
    return (cell * mask).sum(-1)


def mse(g, mu, w2, mask, **_: object) -> torch.Tensor:
    """Unweighted squared error: the contrast case that ignores the noise."""
    return (0.5 * (g - mu) ** 2 * mask).sum(-1)


def chi2(g, mu, w2, mask) -> torch.Tensor:
    """Per-object chi-square ``sum_j m_j (g_j - mu_j)^2 / w_j^2``."""
    return (((g - mu) ** 2 / w2.clamp_min(1e-12)) * mask).sum(-1)


LOSSES: dict[str, ReconstructionLoss] = {
    "gaussian_nll": gaussian_nll,
    "student_t": student_t,
    "mse": mse,
}
"""Name -> reconstruction term. ``loss.name`` in the config selects one.

``weighted_mse`` was removed: with a fixed variance it has exactly the gradients
of ``gaussian_nll``.
"""


def get_loss(name: str) -> ReconstructionLoss:
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; allowed: {sorted(LOSSES)}") from None
