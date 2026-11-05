"""A standard variational autoencoder over the 60-band SED shape.

The model is deliberately plain: a 1-D convolutional encoder along the band
axis (weight sharing between neighbouring filters, matching the physical
overlap), a linear latent posterior, and a decoder that emits the shape vector
directly. Nothing here knows about batch sizes, files or losses.

Three things are knobs, all in :class:`jpas_da.modeling.config.ModelConfig`:

* ``channels`` -- which of ``g``, ``v``, ``mask`` the encoder sees;
* ``hidden`` -- one width per convolutional layer, plus ``kernel_size``;
* the loss and its parameters, selected by name in ``LossConfig``.

Everything else follows from the data: ``n_bands`` comes from the array, the
latent width from the config.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise

import torch
from torch import nn


def _mlp(sizes: list[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for index, (fan_in, fan_out) in enumerate(pairwise(sizes)):
        layers.append(nn.Linear(fan_in, fan_out))
        if index < len(sizes) - 2:
            layers.append(nn.GELU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


@dataclass
class VAEShapes:
    n_bands: int
    n_channels: int
    latent_dim: int


class Encoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_bands: int,
        hidden: tuple[int, ...],
        latent_dim: int,
        kernel_size: int = 5,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd so the band axis keeps its length")
        blocks: list[nn.Module] = []
        fan_in = n_channels
        for width in hidden:
            blocks += [
                nn.Conv1d(fan_in, width, kernel_size, padding=kernel_size // 2),
                nn.GELU(),
            ]
            if dropout > 0:
                blocks.append(nn.Dropout(dropout))
            fan_in = width
        self.conv = nn.Sequential(*blocks)
        # Mean and max pooling together keep the order information a pure mean
        # would discard, without a 60-way flatten.
        self.head = _mlp([2 * hidden[-1], 2 * latent_dim], dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.conv(x)
        pooled = torch.cat([features.mean(-1), features.amax(-1)], dim=-1)
        mu, logvar = self.head(pooled).chunk(2, dim=-1)
        return mu, logvar.clamp(-20.0, 20.0)


class Decoder(nn.Module):
    def __init__(
        self,
        n_bands: int,
        hidden: tuple[int, ...],
        latent_dim: int,
        scale_head: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.n_bands = n_bands
        self.scale_head = scale_head
        self.backbone = _mlp([latent_dim, *hidden], dropout)
        self.shape_out = nn.Linear(hidden[-1], n_bands)
        self.scale_out = nn.Linear(hidden[-1], 1) if scale_head else None

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        features = self.backbone(z)
        shape = self.shape_out(features)
        scale = self.scale_out(features) if self.scale_out is not None else None
        return shape, scale


class VAE(nn.Module):
    """Gaussian-posterior VAE whose decoder emits the SED shape.

    ``forward`` returns ``mu_z``, ``logvar_z``, ``z``, ``mu_g`` (raw decoder
    output) and ``l_mu`` (predicted log brightness, or ``None``).
    """

    def __init__(
        self,
        n_bands: int,
        n_channels: int,
        latent_dim: int,
        hidden: tuple[int, ...],
        kernel_size: int,
        scale_head: bool,
        dropout: float,
    ) -> None:
        """All architectural values are required: defaults live in ModelConfig only."""
        super().__init__()
        if not hidden:
            raise ValueError("hidden must contain at least one width")
        self.shapes = VAEShapes(n_bands, n_channels, latent_dim)
        self.encoder = Encoder(n_channels, n_bands, hidden, latent_dim, kernel_size, dropout)
        self.decoder = Decoder(n_bands, hidden, latent_dim, scale_head, dropout)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | None]:
        mu_z, logvar_z = self.encode(x)
        z = self.reparameterise(mu_z, logvar_z)
        mu_g, l_mu = self.decode(z)
        return {"mu_z": mu_z, "logvar_z": logvar_z, "z": z, "mu_g": mu_g, "l_mu": l_mu}


def normalise_shape(mu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Project the decoder output onto the unit sphere over the measured bands.

    The target ``g`` is unit-norm over ``mask == 1``; scoring a raw decoder
    output would charge it for a radial residual the data cannot distinguish
    from a shape error.
    """
    mu = mu * mask
    norm = mu.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return mu / norm


def build_model(spec: dict) -> VAE:
    """Construct a :class:`VAE` from a config mapping or a checkpoint's spec.

    Both training and inference go through here, so a checkpoint can only be
    restored with the architecture it was trained with.
    """
    return VAE(
        n_bands=int(spec["n_bands"]),
        n_channels=int(spec["n_channels"]),
        latent_dim=int(spec["latent_dim"]),
        hidden=tuple(int(width) for width in spec["hidden"]),
        kernel_size=int(spec["kernel_size"]),
        scale_head=bool(spec["scale_head"]),
        dropout=float(spec.get("dropout", 0.0)),
    )
