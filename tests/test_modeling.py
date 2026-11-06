"""Invariants of the loader, the configuration, the VAE and the loss registry.

These are the properties the training loop relies on: rotations never mix
roles, the encoder input layout is fixed, masked bands carry zero weight, the
config rejects unknown keys, and every registered loss returns one finite value
per object.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import yaml

from jpas_da.modeling import data as model_data
from jpas_da.modeling.config import Config, load_config
from jpas_da.modeling.losses import LOSSES, chi2, get_loss, noise_variance
from jpas_da.modeling.vae import build_model, kl_per_dim


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
N_BANDS, N_NARROW = 60, 54


@pytest.fixture
def batch():
    generator = torch.Generator().manual_seed(0)
    n = 5
    mask = torch.ones(n, N_BANDS)
    mask[0, 0] = 0.0
    g = torch.randn(n, N_BANDS, generator=generator) * mask
    v = (0.1 * g.abs() + 0.05) * mask
    sig_logr = torch.full((n,), 0.1)
    return g, v, mask, sig_logr


def _spec(**overrides):
    spec = {
        "encoder": "conv",
        "n_channels": 3,
        "n_cond": 1,
        "n_bands": N_BANDS,
        "n_narrow": N_NARROW,
        "latent_dim": 3,
        "conv_channels": (4, 8),
        "kernel_size": 5,
        "narrow_features": 16,
        "other_hidden": (8,),
        "hidden": (16, 8),
        "dropout": 0.0,
    }
    spec.update(overrides)
    return spec


def _inputs(n=4, n_channels=3, n_cond=1):
    return (
        torch.randn(n, n_channels, N_NARROW),
        torch.randn(n, n_channels * (N_BANDS - N_NARROW)),
        torch.randn(n, n_cond),
    )


@pytest.mark.parametrize("encoder", ["conv", "mlp"])
@pytest.mark.parametrize("channels", [1, 2, 3])
def test_both_encoders_accept_every_channel_count(encoder, channels):
    model = build_model(_spec(encoder=encoder, n_channels=channels))
    out = model(*_inputs(n_channels=channels))
    assert out["mu_z"].shape == (4, 3)
    assert out["g_hat"].shape == (4, N_BANDS)


def test_conv_encoder_sees_where_a_feature_sits_on_the_band_axis():
    """No global pooling: moving a feature along the narrow axis changes the latent."""
    torch.manual_seed(0)
    model = build_model(_spec()).eval()
    narrow, other, cond = _inputs(n=1)
    narrow.zero_()
    early, late = narrow.clone(), narrow.clone()
    early[0, 0, 10] = 5.0
    late[0, 0, 40] = 5.0
    with torch.no_grad():
        z_early = model(early, other, cond, sample=False)["mu_z"]
        z_late = model(late, other, cond, sample=False)["mu_z"]
    assert not torch.allclose(z_early, z_late, atol=1e-4)


def test_evaluation_without_sampling_is_deterministic():
    model = build_model(_spec())
    inputs = _inputs()
    first = model(*inputs, sample=False)
    second = model(*inputs, sample=False)
    torch.testing.assert_close(first["g_hat"], second["g_hat"])
    torch.testing.assert_close(first["z"], first["mu_z"])


def test_condition_reaches_the_decoder():
    model = build_model(_spec()).eval()
    z = torch.zeros(1, 3)
    low = model.decode(z, torch.tensor([[2.0]]))
    high = model.decode(z, torch.tensor([[8.0]]))
    assert not torch.allclose(low, high)


def test_condition_scale_is_stored_with_the_weights():
    model = build_model(_spec())
    model.set_condition_scale(torch.tensor([[1.0], [3.0], [5.0]]))
    state = model.state_dict()
    restored = build_model(_spec())
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.cond_mean, torch.tensor([3.0]))
    torch.testing.assert_close(restored.cond_scale, torch.tensor([2.0]))


def test_checkpoint_spec_cannot_be_loaded_into_a_different_architecture():
    trained = build_model(_spec(latent_dim=5, hidden=(12,)))
    build_model(_spec(latent_dim=5, hidden=(12,))).load_state_dict(trained.state_dict())
    with pytest.raises(RuntimeError):
        build_model(_spec(latent_dim=5, hidden=(13,))).load_state_dict(trained.state_dict())
    with pytest.raises(RuntimeError):
        build_model(_spec(latent_dim=4, hidden=(12,))).load_state_dict(trained.state_dict())


def test_incomplete_spec_is_rejected():
    spec = _spec()
    del spec["encoder"]
    with pytest.raises(ValueError, match="encoder"):
        build_model(spec)


def test_noise_variance_adds_the_floor_and_the_common_mode(batch):
    g, v, mask, sig_logr = batch
    plain = noise_variance(g, v, mask, sig_logr, eps=0.1, common_mode=False)
    torch.testing.assert_close(plain, (v**2 + 0.01) * mask)
    shared = noise_variance(g, v, mask, sig_logr, eps=0.1, common_mode=True)
    torch.testing.assert_close(shared - plain, (g * 0.1) ** 2 * mask)
    assert shared[0, 0] == 0.0, "masked band must have zero variance"


def test_per_band_floor_broadcasts(batch):
    g, v, mask, sig_logr = batch
    eps = torch.linspace(0.01, 0.2, N_BANDS)
    w2 = noise_variance(g, v, mask, sig_logr, eps=eps, common_mode=False)
    torch.testing.assert_close(w2, (v**2 + eps**2) * mask)


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_every_loss_is_finite_and_per_object(name, batch):
    g, v, mask, sig_logr = batch
    w2 = noise_variance(g, v, mask, sig_logr, eps=0.1)
    values = get_loss(name)(g, torch.randn_like(g), w2, mask)
    assert values.shape == (len(g),)
    assert torch.isfinite(values).all()


def test_gaussian_nll_penalises_a_wrong_shape_more_than_a_right_one(batch):
    g, v, mask, sig_logr = batch
    w2 = noise_variance(g, v, mask, sig_logr, eps=0.1)
    loss_fn = get_loss("gaussian_nll")
    assert loss_fn(g, -g, w2, mask).mean() > loss_fn(g, g, w2, mask).mean()


@pytest.mark.parametrize("name", sorted(LOSSES))
def test_masked_band_contributes_nothing_to_the_loss(name, batch):
    g, v, mask, sig_logr = batch
    w2 = noise_variance(g, v, mask, sig_logr, eps=0.1)
    mu = torch.randn_like(g)
    moved = mu.clone()
    moved[0, 0] = 123.0
    loss_fn = get_loss(name)
    torch.testing.assert_close(loss_fn(g, mu, w2, mask)[0], loss_fn(g, moved, w2, mask)[0])


def test_gaussian_nll_gradient_matches_the_inverse_variance_weighting(batch):
    """With fixed w2 the log term is constant: only the chi2 drives the gradient."""
    g, v, mask, sig_logr = batch
    w2 = noise_variance(g, v, mask, sig_logr, eps=0.1)
    mu = torch.randn_like(g, requires_grad=True)
    get_loss("gaussian_nll")(g, mu, w2, mask).sum().backward()
    grad_nll = mu.grad.clone()
    mu.grad = None
    (0.5 * chi2(g, mu, w2, mask)).sum().backward()
    torch.testing.assert_close(grad_nll, mu.grad)


def test_kl_is_zero_at_the_prior_and_positive_elsewhere():
    zeros = torch.zeros(2, 3)
    torch.testing.assert_close(kl_per_dim(zeros, zeros), zeros)
    assert (kl_per_dim(torch.ones(2, 3), zeros) > 0).all()


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
def test_chi2_ignores_masked_bands_and_counts_measured_ones():
    from jpas_da.modeling.evaluate import chi2_per_object

    g = np.zeros((2, 4))
    g_hat = np.ones((2, 4))
    w2 = np.full((2, 4), 0.25)
    mask = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    g_hat[1, 2:] = 1e6  # masked, must not count
    chi2, dof = chi2_per_object(g, g_hat, w2, mask)
    np.testing.assert_allclose(chi2, [16.0, 8.0])
    np.testing.assert_allclose(dof, [4, 2])


def test_reconstruction_report_bins_cover_every_object():
    from jpas_da.modeling.evaluate import reconstruction_report

    rng = np.random.default_rng(0)
    n = 200
    chi2 = rng.chisquare(50, n)
    dof = np.full(n, 50.0)
    sig_logr = rng.uniform(0.01, 0.5, n)
    report = reconstruction_report(
        chi2, dof, sig_logr, np.repeat([1, 2], n // 2), np.array(["clean"] * n)
    )
    assert sum(b["n"] for b in report["by_snr"].values()) == n
    assert sum(b["n"] for b in report["by_tile"].values()) == n
    assert abs(report["overall"]["chi2_dof_pooled"] - 1.0) < 0.1


def test_latent_report_counts_active_dimensions():
    from jpas_da.modeling.evaluate import latent_report

    kl = np.zeros((10, 4))
    kl[:, 0] = 2.0
    kl[:, 2] = 0.5
    report = latent_report(kl)
    assert report["active_dims"] == 2
    assert report["kl_total"] == pytest.approx(2.5)


# --------------------------------------------------------------------------- #
# training and prediction, end to end on synthetic arrays
# --------------------------------------------------------------------------- #
def test_train_and_predict_run_end_to_end(tmp_path):
    import json

    from jpas_da.modeling import predict, train

    arrays = _arrays(n_per_tile=40)
    arrays["mask"][0, 5] = 0
    data_path = tmp_path / "inputs.npz"
    np.savez(data_path, **arrays)
    overrides = [
        f"data.path={data_path}",
        "data.rotation=1",
        "model.latent_dim=2",
        "model.conv_channels=[4]",
        "model.narrow_features=8",
        "model.other_hidden=[8]",
        "model.hidden=[16]",
        "train.epochs=3",
        "train.batch_size=32",
        "loss.beta_warmup_fraction=0.3",
        f"output.model_path={tmp_path / 'vae.pt'}",
        f"output.metrics_path={tmp_path / 'metrics.json'}",
    ]
    metrics = train.main(config_path=None, overrides=overrides, debug=False, seed=0)
    assert (tmp_path / "vae.pt").exists() and (tmp_path / "vae_last.pt").exists()
    saved = json.loads((tmp_path / "metrics.json").read_text())
    assert saved["rotation"] == 1
    assert 1 <= saved["best_epoch"] <= 3
    assert len(saved["history"][-1]["val_kl_per_dim"]) == 2
    assert np.isfinite(metrics["test"]["reconstruction"]["overall"]["chi2_dof_pooled"])

    table = predict.main(model_path=tmp_path / "vae.pt", data_path=data_path, device_name="cpu")
    assert len(table) == len(arrays["g"])
    assert {"z_0", "z_1", "chi2", "role"} <= set(table.columns)
    assert (table.loc[arrays["holdout"], "role"] == "holdout").all()
    expected = model_data.roles(arrays, 1).astype(str)
    np.testing.assert_array_equal(table["role"].to_numpy(), expected)


# --------------------------------------------------------------------------- #
# baselines
# --------------------------------------------------------------------------- #
def test_weighted_pca_recovers_a_noiseless_low_rank_signal():
    from jpas_da.modeling.baselines import WeightedPCA

    rng = np.random.default_rng(0)
    basis = rng.normal(size=(2, 20))
    g = 1.0 + rng.normal(size=(300, 2)) @ basis
    w2 = rng.uniform(0.01, 1.0, size=g.shape)  # unequal weights must not matter
    mask = np.ones_like(g)
    mask[::5, 3] = 0
    model = WeightedPCA(2, n_iter=50).fit(g, w2, mask)
    np.testing.assert_allclose(model.reconstruct(g, w2, mask), g, atol=1e-5)


def test_weighted_pca_downweights_noisy_cells():
    """A huge error on one cell must not pull the fit towards that cell."""
    from jpas_da.modeling.baselines import WeightedPCA

    rng = np.random.default_rng(1)
    g = np.outer(rng.normal(size=200), np.linspace(1, 2, 10))
    corrupted = g.copy()
    corrupted[0, 0] += 1e3
    w2 = np.full(g.shape, 0.01)
    w2[0, 0] = 1e8
    model = WeightedPCA(1, n_iter=50).fit(corrupted, w2, np.ones_like(g))
    clean_rows = model.reconstruct(g[1:], w2[1:], np.ones_like(g[1:]))
    np.testing.assert_allclose(clean_rows, g[1:], atol=1e-3)


def test_asinh_pca_at_full_rank_reproduces_the_measured_shape():
    from jpas_da.modeling.baselines import AsinhPCA

    rng = np.random.default_rng(2)
    g = rng.normal(1.0, 0.5, size=(100, 8))
    v = np.full_like(g, 0.2)
    mask = np.ones_like(g)
    logr = rng.normal(4.0, 1.0, 100)
    model = AsinhPCA(8).fit(g, v, mask, logr)
    np.testing.assert_allclose(model.reconstruct(g, v, mask, logr), g, atol=1e-8)
