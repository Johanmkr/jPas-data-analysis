"""Reconstruction terms, as a registry keyed by name.

A loss here owns exactly one thing: how a decoder prediction ``mu`` is scored
against the measured shape ``g`` given the propagated error ``v`` and the mask.
The KL term and the brightness head are handled by the training loop, because
they are properties of the variational model rather than of the noise model.

Every function has the same signature and returns one value per object, summed
over bands::

    loss(g, v, mask, mu, s_floor) -> Tensor   # shape (N,)

``mask`` is 1/0 per band, ``v`` is zero where the mask is zero, and ``s_floor``
is the fixed per-band additive error floor derived on the training fold.

Adding a noise model is adding one entry to :data:`LOSSES` plus, if it has
parameters, one field in ``LossConfig``.
"""

from typing import Protocol

import torch


class ReconstructionLoss(Protocol):
    """Score a decoder prediction; returns one value per object."""

    def __call__(
        self,
        g: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        s_floor: torch.Tensor,
        **kwargs: object,
    ) -> torch.Tensor: ...


def _weights(v: torch.Tensor, s_floor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-cell variance ``w^2 = v^2 + s^2``, zero on masked bands."""
    return (v**2 + s_floor**2) * mask


def gaussian_nll(
    g: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    s_floor: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    """Gaussian negative log-likelihood with known, per-cell variance.

    Includes the ``0.5 log w^2`` noise-normalisation term: without it the band
    budget is set by noise rather than by information.
    """
    w2 = _weights(v, s_floor, mask)
    w2 = w2.clamp_min(1e-12) * mask
    residual = (g - mu) ** 2 / w2.clamp_min(1e-12)
    cell = 0.5 * (residual + torch.log(w2.clamp_min(1e-12)))
    return (cell * mask).sum(-1)


def student_t(
    g: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    s_floor: torch.Tensor,
    *,
    dof: float = 5.0,
    **_: object,
) -> torch.Tensor:
    """Heavy-tailed alternative to :func:`gaussian_nll`, same scale ``w``.

    Use it to test whether the latent geometry is being set by gross outliers:
    if the embedding is unchanged, the Gaussian was not outlier-driven.
    """
    if dof <= 0:
        raise ValueError("student_t dof must be positive")
    w2 = (_weights(v, s_floor, mask) * mask).clamp_min(1e-12)
    residual = (g - mu) ** 2 / w2
    cell = 0.5 * (dof + 1.0) * torch.log1p(residual / dof)
    return (cell * mask).sum(-1)


def weighted_mse(
    g: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    s_floor: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    """Inverse-variance weighted squared error, no log term.

    Kept as the documented contrast case: it allocates the loss budget in
    proportion to the noise variance, so the shallowest bands dominate.
    """
    w2 = (_weights(v, s_floor, mask) * mask).clamp_min(1e-12)
    return (0.5 * (g - mu) ** 2 / w2 * mask).sum(-1)


def mse(
    g: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    mu: torch.Tensor,
    s_floor: torch.Tensor,
    **_: object,
) -> torch.Tensor:
    """Unweighted squared error; every band counts the same."""
    return (0.5 * (g - mu) ** 2 * mask).sum(-1)


LOSSES: dict[str, ReconstructionLoss] = {
    "gaussian_nll": gaussian_nll,
    "student_t": student_t,
    "weighted_mse": weighted_mse,
    "mse": mse,
}
"""Name -> reconstruction term. ``loss.name`` in the config selects one."""


def get_loss(name: str) -> ReconstructionLoss:
    try:
        return LOSSES[name]
    except KeyError:
        raise ValueError(f"unknown loss {name!r}; allowed: {sorted(LOSSES)}") from None
