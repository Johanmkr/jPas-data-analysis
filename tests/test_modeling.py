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
from jpas_da.modeling.vae import build_model, normalise_shape

N_BANDS = 12


# --------------------------------------------------------------------------- #
# loader
# --------------------------------------------------------------------------- #
def _arrays(n_per_tile: int = 10) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    tiles = np.repeat([2241, 2243, 2406, 2470], n_per_tile)
    n = len(tiles)
    holdout = np.zeros(n, dtype=bool)
    holdout[::7] = True
    quality = np.array(["clean"] * n)
    quality[3::11] = "severe"
    return {
        "g": rng.normal(size=(n, 60)).astype(np.float32),
        "v": rng.uniform(0.01, 0.1, size=(n, 60)).astype(np.float32),
        "mask": np.ones((n, 60), dtype=np.int8),
        "logr": rng.normal(5.0, 1.0, n).astype(np.float32),
        "sig_logr": rng.uniform(0.05, 0.3, n).astype(np.float32),
        "object_id": np.array([f"{t}-{i}" for i, t in enumerate(tiles)]),
        "tile_id": tiles,
        "holdout": holdout,
        "quality": quality,
    }


@pytest.mark.parametrize("rotation", range(4))
def test_split_indices_are_disjoint_and_skip_holdout_and_severe(rotation):
    arrays = _arrays()
    parts = model_data.split_indices(arrays, rotation)
    seen = np.concatenate([parts[r] for r in ("train", "val", "test")])
    assert len(seen) == len(set(seen.tolist())), "a row has two roles"
    assert not arrays["holdout"][seen].any()
    assert not (arrays["quality"][seen] == "severe").any()
    np.testing.assert_array_equal(parts["holdout"], np.flatnonzero(arrays["holdout"]))
    train_tiles = set(arrays["tile_id"][parts["train"]].tolist())
    assert train_tiles.isdisjoint(arrays["tile_id"][parts["test"]].tolist())
    assert train_tiles.isdisjoint(arrays["tile_id"][parts["val"]].tolist())


def test_model_inputs_split_the_band_axis_into_narrow_and_other():
    arrays = _arrays()
    inputs = model_data.model_inputs(arrays, ("g", "mask"), ("logr", "sig_logr"))
    n = len(arrays["g"])
    assert inputs["narrow"].shape == (n, 2, 54)
    assert inputs["other"].shape == (n, 2 * 6)
    assert inputs["cond"].shape == (n, 2)
    np.testing.assert_array_equal(inputs["narrow"][:, 0], arrays["g"][:, :54])
    # band-major: (g, mask) of uJAVA first, then of uJPAS, ...
    uJAVA = model_data.OTHER_INDEX[0]
    assert inputs["other"][0, 0] == arrays["g"][0, uJAVA]
    assert inputs["other"][0, 1] == arrays["mask"][0, uJAVA]


def test_other_bands_are_the_six_off_grid_filters():
    from jpas_da.bands import BAND_NAMES

    names = [BAND_NAMES[i] for i in model_data.OTHER_INDEX]
    assert names == ["uJAVA", "uJPAS", "gSDSS", "rSDSS", "iSDSS", "J1007"]
    assert len(model_data.NARROW_INDEX) == 54


def test_conditions_must_include_brightness():
    with pytest.raises(ValueError, match="logr"):
        model_data.model_inputs(_arrays(), conditions=("sig_logr",))


def test_load_reports_missing_keys(tmp_path):
    path = tmp_path / "inputs.npz"
    arrays = _arrays()
    del arrays["quality"]
    np.savez(path, **arrays)
    with pytest.raises(ValueError, match="quality"):
        model_data.load(path)


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


def _spec(**overrides):
    spec = {
        "n_bands": N_BANDS,
        "n_channels": 3,
        "latent_dim": 3,
        "hidden": (16, 8),
        "kernel_size": 5,
        "scale_head": True,
        "dropout": 0.0,
    }
    spec.update(overrides)
    return spec


def test_encoder_accepts_every_channel_subset():
    for channels in (("g",), ("g", "v"), ("g", "v", "mask")):
        model = build_model(_spec(n_channels=len(channels)))
        out = model(torch.zeros(4, len(channels), N_BANDS))
        assert out["mu_z"].shape == (4, 3)
        assert out["mu_g"].shape == (4, N_BANDS)


def test_checkpoint_spec_cannot_be_loaded_into_a_different_architecture():
    """The spec alone decides the architecture, so a mismatched rebuild fails."""
    trained = build_model(_spec(kernel_size=3, latent_dim=5, hidden=(12,)))
    matching = build_model(_spec(kernel_size=3, latent_dim=5, hidden=(12,)))
    matching.load_state_dict(trained.state_dict())

    wrong_width = build_model(_spec(kernel_size=3, latent_dim=5, hidden=(13,)))
    with pytest.raises(RuntimeError):
        wrong_width.load_state_dict(trained.state_dict())

    wrong_latent = build_model(_spec(kernel_size=3, latent_dim=4, hidden=(12,)))
    with pytest.raises(RuntimeError):
        wrong_latent.load_state_dict(trained.state_dict())


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
