"""Conditional VAE over the 60-band SED, with a two-branch encoder.

Inputs (built by :func:`jpas_da.modeling.data.model_inputs`):

* ``narrow`` ``(N, C, 54)`` -- the narrow bands on the evenly spaced grid;
* ``other`` ``(N, C * 6)`` -- uJAVA, the four broad bands and J1007;
* ``cond`` ``(N, K)`` -- ``log r_bb`` (and optionally ``sig_logr``).

Encoder (``kind="conv"``): 1-D convolutions along the narrow-band axis, then a
**flatten**, not a pooling: where a feature sits in wavelength is mostly the
redshift signal, and global pooling throws it away. The six off-grid filters
are not convolved -- they are not wavelength neighbours of each other or of
J0910 -- but go through their own small MLP together with the conditions. Both
branches are concatenated and mapped to ``(mu, logvar)``.

``kind="mlp"`` is the ablation: every input flattened into one MLP.

Decoder: ``[z, cond] -> MLP -> 60`` values of ``g`` in canonical band order. The
brightness is a *condition*, given to encoder and decoder and never predicted,
so the latent describes the SED shape given the brightness.

The conditions are standardised inside the model with a mean and scale fitted
on the training rows (:meth:`VAE.set_condition_scale`) and stored in the
checkpoint as buffers.

Encoder trunk, decoder trunk and latent head are separate modules, so a
non-Euclidean latent only has to replace the head and the KL.
"""

from __future__ import annotations

from itertools import pairwise

import torch
from torch import nn

ENCODERS = ("conv", "mlp")


def mlp(sizes: list[int], dropout: float = 0.0, final_activation: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    n_layers = len(sizes) - 1
    for index, (fan_in, fan_out) in enumerate(pairwise(sizes)):
        layers.append(nn.Linear(fan_in, fan_out))
        if index < n_layers - 1 or final_activation:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class ConvTrunk(nn.Module):
    """Narrow bands by convolution, off-grid bands and conditions by MLP, fused."""

    def __init__(
        self,
        n_channels: int,
        n_narrow: int,
        n_other: int,
        n_cond: int,
        conv_channels: tuple[int, ...],
        kernel_size: int,
        narrow_features: int,
        other_hidden: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the band axis keeps its length")
        blocks: list[nn.Module] = []
        fan_in = n_channels
        for width in conv_channels:
            blocks += [nn.Conv1d(fan_in, width, kernel_size, padding=kernel_size // 2), nn.GELU()]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            fan_in = width
        self.conv = nn.Sequential(*blocks)
        self.narrow_out = nn.Sequential(
            nn.Flatten(), nn.Linear(fan_in * n_narrow, narrow_features), nn.GELU()
        )
        self.other = mlp([n_other + n_cond, *other_hidden], dropout, final_activation=True)
        self.out_features = narrow_features + other_hidden[-1]

    def forward(self, narrow: torch.Tensor, other: torch.Tensor, cond: torch.Tensor):
        a = self.narrow_out(self.conv(narrow))
        b = self.other(torch.cat([other, cond], dim=-1))
        return torch.cat([a, b], dim=-1)


class FlatTrunk(nn.Module):
    """Ablation: everything flattened, no convolution."""

    def __init__(self, n_inputs: int) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.out_features = n_inputs

    def forward(self, narrow: torch.Tensor, other: torch.Tensor, cond: torch.Tensor):
        return torch.cat([self.flatten(narrow), other, cond], dim=-1)


class GaussianHead(nn.Module):
    """Features -> diagonal Gaussian posterior ``(mu, logvar)``."""

    def __init__(self, n_features: int, hidden: tuple[int, ...], latent_dim: int, dropout: float):
        super().__init__()
        self.net = mlp([n_features, *hidden, 2 * latent_dim], dropout)

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mu, logvar = self.net(features).chunk(2, dim=-1)
        return mu, logvar.clamp(-20.0, 20.0)


class VAE(nn.Module):
    """Conditional Gaussian-posterior VAE.

    ``forward`` returns ``mu_z``, ``logvar_z``, ``z`` (a sample, or ``mu_z`` when
    ``sample=False``) and ``g_hat`` (``(N, n_bands)``).
    """

    def __init__(
        self,
        *,
        encoder: str,
        n_channels: int,
        n_cond: int,
        n_bands: int,
        n_narrow: int,
        latent_dim: int,
        conv_channels: tuple[int, ...],
        kernel_size: int,
        narrow_features: int,
        other_hidden: tuple[int, ...],
        hidden: tuple[int, ...],
        dropout: float,
    ) -> None:
        """All architectural values are required: defaults live in ModelConfig only."""
        super().__init__()
        if encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {encoder!r}; expected one of {ENCODERS}")
        if not hidden:
            raise ValueError("hidden must contain at least one width")
        n_other = n_channels * (n_bands - n_narrow)
        if encoder == "conv":
            if not conv_channels or not other_hidden:
                raise ValueError("the conv encoder needs conv_channels and other_hidden")
            self.trunk = ConvTrunk(
                n_channels,
                n_narrow,
                n_other,
                n_cond,
                conv_channels,
                kernel_size,
                narrow_features,
                other_hidden,
                dropout,
            )
        else:
            self.trunk = FlatTrunk(n_channels * n_narrow + n_other + n_cond)
        self.head = GaussianHead(self.trunk.out_features, hidden, latent_dim, dropout)
        self.decoder = mlp([latent_dim + n_cond, *reversed(hidden), n_bands], dropout)
        self.latent_dim = latent_dim
        self.register_buffer("cond_mean", torch.zeros(n_cond))
        self.register_buffer("cond_scale", torch.ones(n_cond))

    def set_condition_scale(self, cond: torch.Tensor) -> None:
        """Fix the condition standardisation from the training rows."""
        self.cond_mean.copy_(cond.mean(0))
        self.cond_scale.copy_(cond.std(0).clamp_min(1e-6))

    def _cond(self, cond: torch.Tensor) -> torch.Tensor:
        return (cond - self.cond_mean) / self.cond_scale

    def encode(self, narrow, other, cond) -> tuple[torch.Tensor, torch.Tensor]:
        return self.head(self.trunk(narrow, other, self._cond(cond)))

    def decode(self, z: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([z, self._cond(cond)], dim=-1))

    def forward(self, narrow, other, cond, sample: bool = True) -> dict[str, torch.Tensor]:
        mu_z, logvar_z = self.encode(narrow, other, cond)
        z = mu_z + torch.exp(0.5 * logvar_z) * torch.randn_like(mu_z) if sample else mu_z
        return {"mu_z": mu_z, "logvar_z": logvar_z, "z": z, "g_hat": self.decode(z, cond)}


def kl_per_dim(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL(q || N(0, 1)) per object and latent dimension, ``(N, d)``."""
    return 0.5 * (mu**2 + torch.exp(logvar) - 1.0 - logvar)


SPEC_KEYS = (
    "encoder",
    "n_channels",
    "n_cond",
    "n_bands",
    "n_narrow",
    "latent_dim",
    "conv_channels",
    "kernel_size",
    "narrow_features",
    "other_hidden",
    "hidden",
    "dropout",
)


def build_model(spec: dict) -> VAE:
    """Construct a :class:`VAE` from a config mapping or a checkpoint's spec.

    Training and inference both go through here, so a checkpoint can only be
    restored with the architecture it was trained with.
    """
    missing = set(SPEC_KEYS) - spec.keys()
    if missing:
        raise ValueError(f"model spec is missing {sorted(missing)}")
    return VAE(
        encoder=str(spec["encoder"]),
        n_channels=int(spec["n_channels"]),
        n_cond=int(spec["n_cond"]),
        n_bands=int(spec["n_bands"]),
        n_narrow=int(spec["n_narrow"]),
        latent_dim=int(spec["latent_dim"]),
        conv_channels=tuple(int(w) for w in spec["conv_channels"]),
        kernel_size=int(spec["kernel_size"]),
        narrow_features=int(spec["narrow_features"]),
        other_hidden=tuple(int(w) for w in spec["other_hidden"]),
        hidden=tuple(int(w) for w in spec["hidden"]),
        dropout=float(spec["dropout"]),
    )
