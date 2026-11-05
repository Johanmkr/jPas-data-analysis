"""Invariants of the model-ready tensors, the VAE and the loss registry.

These are the properties the training loop relies on: the shape vector is
unit-norm over measured bands only, masked bands carry zero weight, the config
rejects unknown keys, and every registered loss returns one finite value per
object.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from jpas_da.modeling import data as model_data
from jpas_da.modeling.config import Config, load_config
from jpas_da.modeling.losses import LOSSES, get_loss
from jpas_da.modeling.vae import VAE, normalise_shape

N_BANDS = 12


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def _synthetic(n_rows: int = 8, n_bands: int = N_BANDS, masked=None):
    rng = np.random.default_rng(0)
    flux = rng.normal(scale=10.0, size=(n_rows, n_bands))
    sigma = np.abs(flux) * 0.2 + 0.01
    mask = np.ones((n_rows, n_bands), dtype=np.int8)
    if masked is not None:
        for row, band in masked:
            mask[row, band] = 0
            sigma[row, band] = np.nan
    return flux, sigma, mask


def test_shape_is_unit_norm_over_measured_bands():
    flux, sigma, mask = _synthetic(masked=[(0, 3), (1, 2)])
    out = model_data.normalise(flux, sigma, mask)
    norms = np.sqrt((out["g"].astype(np.float64) ** 2 * out["mask"]).sum(axis=1))
    np.testing.assert_allclose(norms, 1.0, rtol=1e-5)


def test_masked_band_has_zero_shape_and_zero_error():
    flux, sigma, mask = _synthetic(masked=[(0, 3)])
    out = model_data.normalise(flux, sigma, mask)
    assert out["mask"][0, 3] == 0
    assert out["g"][0, 3] == 0.0
    assert out["v"][0, 3] == 0.0


def test_masked_band_flux_does_not_move_the_norm():
    """A band with no quoted error is excluded from the norm, not merely zeroed."""
    flux, sigma, mask = _synthetic(masked=[(0, 5)])
    baseline = model_data.normalise(flux, sigma, mask)
    loud = flux.copy()
    loud[0, 5] = 1e6
    perturbed = model_data.normalise(loud, sigma, mask)
    np.testing.assert_allclose(perturbed["logr"][0], baseline["logr"][0], rtol=1e-6)
    np.testing.assert_allclose(
        perturbed["g"][0] * perturbed["mask"][0],
        baseline["g"][0] * baseline["mask"][0],
        rtol=1e-6,
    )


def test_sigma_scales_with_the_multiplicative_floor():
    flux = np.full((2, N_BANDS), 1.0)
    sigma = np.zeros_like(flux)
    mask = np.ones_like(flux, dtype=np.int8)
    quiet = model_data.normalise(flux, sigma, mask, floor_frac=0.0)
    floored = model_data.normalise(flux, sigma, mask, floor_frac=0.026)
    assert quiet["v"].max() == 0.0
    assert floored["v"].min() > 0.0


def test_zero_norm_object_is_rejected():
    flux = np.zeros((1, N_BANDS))
    sigma = np.ones((1, N_BANDS))
    mask = np.ones((1, N_BANDS), dtype=np.int8)
    with pytest.raises(ValueError, match="zero masked norm"):
        model_data.normalise(flux, sigma, mask)


def test_brightness_error_follows_the_propagated_variance():
    flux, sigma, mask = _synthetic()
    out = model_data.normalise(flux, sigma, mask)
    expected = np.sqrt((out["g"] ** 2 * out["v"] ** 2 * out["mask"]).sum(axis=1))
    np.testing.assert_allclose(out["sig_logr"], expected, rtol=1e-5)


# --------------------------------------------------------------------------- #
# configuration
# --------------------------------------------------------------------------- #
def test_defaults_are_valid():
    config = Config().validate()
    assert config.model.channels == ("g", "v", "mask")
    assert config.loss.name in LOSSES


def test_roundtrip_through_yaml_preserves_values(tmp_path):
    original = Config()
    original.train.epochs = 3
    original.model.hidden = (32, 16)
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(original.to_dict()))
    loaded = load_config(path)
    assert loaded.train.epochs == 3
    assert loaded.model.hidden == (32, 16)
    assert loaded.loss.beta == original.loss.beta


def test_unknown_key_is_rejected(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("model:\n  widht: 32\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_config(path)


def test_unknown_loss_is_rejected():
    config = Config()
    config.loss.name = "not_a_loss"
    with pytest.raises(ValueError, match="unknown loss"):
        config.validate()


def test_unknown_channel_is_rejected():
    config = Config()
    config.model.channels = ("g", "redshift")
    with pytest.raises(ValueError, match="unknown channels"):
        config.validate()


def test_cli_overrides_beat_the_file(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("train:\n  epochs: 7\n")
    loaded = load_config(path, ["train.epochs=2", "model.hidden=[8]"])
    assert loaded.train.epochs == 2
    assert loaded.model.hidden == (8,)


def test_even_kernel_is_rejected():
    config = Config()
    config.model.kernel_size = 4
    with pytest.raises(ValueError, match="odd"):
        config.validate()


# --------------------------------------------------------------------------- #
# model and losses
# --------------------------------------------------------------------------- #
@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(0)
    n = 5
    g = torch.randn(n, N_BANDS, generator=generator)
    mask = torch.ones(n, N_BANDS)
    mask[0, 0] = 0.0
    g = g * mask
    g = g / g.norm(dim=-1, keepdim=True)
    v = 0.1 * g.abs() + 0.01
    v = v * mask
    s_floor = torch.full((N_BANDS,), 0.02)
    return g, v, mask, s_floor


def test_encoder_accepts_every_channel_subset():
    for channels in (("g",), ("g", "v"), ("g", "v", "mask")):
        model = VAE(n_bands=N_BANDS, n_channels=len(channels), latent_dim=3, hidden=(16, 8))
        out = model(torch.zeros(4, len(channels), N_BANDS))
        assert out["mu_z"].shape == (4, 3)
        assert out["mu_g"].shape == (4, N_BANDS)


def test_normalised_shape_lies_on_the_sphere_over_the_mask(batch):
    g, _, mask, _ = batch
    mu = normalise_shape(torch.randn_like(g), mask)
    norms = (mu * mask).norm(dim=-1)
    torch.testing.assert_close(norms, torch.ones_like(norms))


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_every_loss_is_finite_and_per_object(name, batch):
    g, v, mask, s_floor = batch
    mu = normalise_shape(torch.randn_like(g), mask)
    values = get_loss(name)(g, v, mask, mu, s_floor)
    assert values.shape == (len(g),)
    assert torch.isfinite(values).all()


def test_gaussian_nll_penalises_a_wrong_shape_more_than_a_right_one(batch):
    g, v, mask, s_floor = batch
    loss_fn = get_loss("gaussian_nll")
    perfect = loss_fn(g, v, mask, g, s_floor).mean()
    wrong = loss_fn(g, v, mask, -g, s_floor).mean()
    assert wrong > perfect


def test_masked_band_contributes_nothing_to_the_loss(batch):
    g, v, mask, s_floor = batch
    loss_fn = get_loss("gaussian_nll")
    mu = normalise_shape(torch.randn_like(g), mask)
    baseline = loss_fn(g, v, mask, mu, s_floor)
    perturbed = mu.clone()
    perturbed[0, 0] = 123.0
    perturbed = perturbed * mask
    changed = loss_fn(g, v, mask, perturbed, s_floor)
    torch.testing.assert_close(baseline[0], changed[0])


def test_noise_floor_is_per_band_and_positive(batch):
    from jpas_da.modeling.train import noise_floor

    _, v, mask, _ = batch
    floor = noise_floor(v, mask, percentile=50.0)
    assert floor.shape == (N_BANDS,)
    assert torch.isfinite(floor).all()
    assert (floor > 0).all()
