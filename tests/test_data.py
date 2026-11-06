"""Invariants of the raw -> interim -> representation pipeline.

These are cheap and run offline on small synthetic frames, plus one integration
test that exercises the real export behind a skip when it is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jpas_da import asinh, features
from jpas_da.bands import BAND_NAMES, BLOCK_ORDER, N_BANDS, select
from jpas_da.config import RAW_DATA_DIR
from jpas_da.dataset import (
    ERROR_COLUMN,
    METADATA_COLUMNS,
    build_photometry,
    clean_metadata,
    drop_duplicate_sources,
    label_flags,
    load_raw,
    quality_class,
)
from jpas_da.splits import assign_roles, rotation_tiles

RAW_FILE = RAW_DATA_DIR / "miniJpas_fluxes_errors_redshifts_full.csv"


# --------------------------------------------------------------------------- #
# bands
# --------------------------------------------------------------------------- #
def test_band_table_is_60_unique_bands():
    assert N_BANDS == 60
    assert len(set(BAND_NAMES)) == 60
    assert sum(len(select(block)) for block in BLOCK_ORDER) == 60
    assert select("narrow")[0] == "J0378"
    assert select("narrow")[-1] == "J0910"
    assert len(select("narrow")) == 54


def test_narrow_bands_form_a_uniform_grid():
    """The 54 narrow filters are a 10 A grid, except 378 -> 390 nm (12 A)."""
    wavelengths = np.array([float(b[1:]) for b in select("narrow")])
    steps = np.diff(wavelengths)
    assert set(np.unique(steps)) == {10.0, 12.0}
    assert steps[0] == 12.0, "only the first step (J0378 -> J0390) may differ"
    assert (steps[1:] == 10.0).all()


def test_error_columns_follow_the_band_order():
    assert [ERROR_COLUMN[b] for b in BAND_NAMES] == [f"e{b}" for b in BAND_NAMES]


# --------------------------------------------------------------------------- #
# synthetic raw frames
# --------------------------------------------------------------------------- #
def _synthetic_raw(n_rows: int = 6, *, undefined_pixel=None) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    flux = rng.normal(10.0, 3.0, size=(n_rows, N_BANDS))
    relerr = rng.uniform(0.02, 0.5, size=(n_rows, N_BANDS))
    if undefined_pixel is not None:
        row, band = undefined_pixel
        relerr[row, band] = 0.0
    data = pd.DataFrame(flux, columns=list(BAND_NAMES))
    for b in BAND_NAMES:
        idx = list(BAND_NAMES).index(b)
        data[ERROR_COLUMN[b]] = relerr[:, idx]
    metadata = {
        "tile_id": [2241] * n_rows,
        "number": np.arange(1, n_rows + 1),
        "ALPHA_J2000": rng.uniform(213, 216, n_rows),
        "DELTA_J2000": rng.uniform(52, 54, n_rows),
        "MAGrSDSS": rng.uniform(18, 25, n_rows),
        "flag_rSDSS": 0,
        "mask_flag_rSDSS": 0,
        "SEtractor": rng.uniform(0, 1, n_rows),
        "photoCLASSsdss": 3.0,
        "spCLASSsdss": "GALAXY",
        "zsp": rng.uniform(0.05, 0.9, n_rows),
        "e_zsp": 0.001,
        "f_zsp": 4.0,
        "zph": rng.uniform(0.05, 0.9, n_rows),
        "e_zph": 0.09,
        "avg_zph": rng.uniform(0.05, 0.9, n_rows),
        "angDist": rng.uniform(0, 1, n_rows),
    }
    for column, values in metadata.items():
        data[column] = values
    return data.loc[:, [*metadata, *BAND_NAMES, *[ERROR_COLUMN[b] for b in BAND_NAMES]]]


def _columns() -> list[str]:
    return list(METADATA_COLUMNS)


# --------------------------------------------------------------------------- #
# photometry
# --------------------------------------------------------------------------- #
def test_sigma_is_absolute_and_matches_flux_times_relative_error():
    data = _synthetic_raw()
    flux, sigma, mask = build_photometry(data)
    band = "J0450"
    expected = np.abs(data[band].to_numpy() * data[ERROR_COLUMN[band]].to_numpy())
    np.testing.assert_allclose(sigma[f"sigma_{band}"].to_numpy(), expected, rtol=1e-12)
    assert (mask[f"mask_{band}"].to_numpy() == 1).all()


def test_undefined_relative_error_becomes_nan_sigma_and_zero_mask():
    data = _synthetic_raw(undefined_pixel=(2, 7))
    band = BAND_NAMES[7]
    flux, sigma, mask = build_photometry(data)
    assert mask[f"mask_{band}"].to_numpy()[2] == 0
    assert np.isnan(sigma[f"sigma_{band}"].to_numpy()[2])
    assert np.isfinite(flux[f"flux_{band}"].to_numpy()[2]), "flux itself must survive"


def test_sigma_stays_positive_even_for_negative_flux():
    data = _synthetic_raw()
    data["J0500"] = -data["J0500"]
    _, sigma, _ = build_photometry(data)
    values = sigma["sigma_J0500"].to_numpy()
    assert (values > 0).all()


def test_mask_and_sigma_never_disagree():
    data = _synthetic_raw(undefined_pixel=(1, 3))
    _, sigma, mask = build_photometry(data)
    defined_sigma = np.isfinite(sigma.to_numpy())
    masked_in = mask.to_numpy() == 1
    assert np.array_equal(defined_sigma, masked_in)


# --------------------------------------------------------------------------- #
# metadata hygiene
# --------------------------------------------------------------------------- #
def test_sentinels_become_nan():
    data = _synthetic_raw()
    data.loc[0, "zph"] = -9999.0
    data.loc[0, "e_zph"] = -9999.0
    data.loc[1, "zsp"] = -0.001
    data.loc[2, "MAGrSDSS"] = 99.0
    meta = clean_metadata(data)
    assert np.isnan(meta.loc[0, "zph"])
    assert np.isnan(meta.loc[0, "e_zph"])
    assert np.isnan(meta.loc[1, "zsp"])
    assert np.isnan(meta.loc[2, "MAGrSDSS"])
    assert meta.loc[3, "zph"] == pytest.approx(data.loc[3, "zph"])


def test_holdout_is_exactly_the_spectroscopic_objects():
    data = _synthetic_raw()
    data["spCLASSsdss"] = data["spCLASSsdss"].astype(object)
    data.loc[[0, 1], "spCLASSsdss"] = np.nan  # photometric-only matches
    data["f_zsp"] = 0.0
    data.loc[2, "f_zsp"] = 4.0  # ZWARNING set: ambiguous fit
    flags = label_flags(clean_metadata(data))
    assert flags["holdout"].tolist() == [False, False, True, True, True, True]
    assert not flags.loc[2, "zsp_reliable"], "a ZWARNING redshift was marked reliable"
    assert flags.loc[3, "zsp_reliable"]
    assert not flags.loc[~flags["holdout"], "zsp_reliable"].any()


def test_quality_class_follows_the_flag_bits():
    flags = np.array([0, 1, 2, 3, 4, 16, 0, 0, 0, 1])
    mask_flags = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 1])
    assert quality_class(flags, mask_flags).tolist() == [
        "clean",
        "blend",
        "blend",
        "blend",
        "severe",  # saturated
        "severe",  # corrupted aperture
        "window",
        "severe",  # bright star
        "severe",  # bright star outranks window
        "window",  # window outranks blend
    ]


def test_clipped_relative_error_is_masked():
    data = _synthetic_raw()
    data.loc[0, ERROR_COLUMN["J0500"]] = 65504.0
    flux, sigma, mask = build_photometry(data)
    assert mask.loc[0, "mask_J0500"] == 0
    assert np.isnan(sigma.loc[0, "sigma_J0500"])


def test_drop_duplicate_sources_keeps_the_tightest_sdss_match():
    data = _synthetic_raw(n_rows=3)
    duplicate = data.loc[[1]].copy()
    duplicate["angDist"] = 0.001  # tighter match than the original row 1
    duplicate["zph"] = 0.5
    stacked = pd.concat([data, duplicate], ignore_index=True)
    assert len(stacked) == 4 and len(data) == 3

    deduped = drop_duplicate_sources(stacked)
    assert len(deduped) == 3
    kept = deduped.loc[deduped["number"] == 2].iloc[0]
    assert kept["angDist"] == pytest.approx(0.001)
    assert kept["zph"] == pytest.approx(0.5)
    # fluxes must be identical across the duplicate, so nothing photometric moved
    band = "J0450"
    assert kept[band] == pytest.approx(data.loc[1, band])


# --------------------------------------------------------------------------- #
# asinh (baseline representation)
# --------------------------------------------------------------------------- #
def test_asinh_roundtrip_recovers_every_flux():
    rng = np.random.default_rng(1)
    flux = rng.normal(0.0, 50.0, size=(20, N_BANDS))
    scale = np.full(N_BANDS, 7.0)
    np.testing.assert_allclose(asinh.inverse(asinh.transform(flux, scale), scale), flux)


def test_asinh_is_linear_in_the_noise_and_logarithmic_in_the_signal():
    scale = np.full(N_BANDS, 10.0)
    faint = np.array([[0.5] * N_BANDS])
    bright = np.array([[1000.0] * N_BANDS])
    np.testing.assert_allclose(asinh.transform(faint, scale), faint / scale, rtol=1e-3)
    np.testing.assert_allclose(
        asinh.transform(bright, scale), np.log(2 * bright / scale), rtol=1e-5
    )


def test_negative_and_zero_flux_are_representable_and_finite():
    scale = np.full(N_BANDS, 10.0)
    flux = np.array([[-50.0, 0.0, 7.0] * 20])
    x = asinh.transform(flux, scale)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(asinh.inverse(x, scale), flux, rtol=1e-12)


def test_propagated_asinh_error_is_flat_once_the_source_dominates():
    """With ``sigma_f = k|f| + floor``, asinh makes ``sigma_x -> k``."""
    scale = np.full(N_BANDS, 10.0)
    flux = np.array([[100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0] * 10])
    errors = asinh.error(flux, 0.4 * np.abs(flux) + 1.0, scale)
    assert errors.max() / errors.min() < 1.05
    assert np.allclose(errors, 0.4, rtol=0.05)


def test_band_scales_use_only_the_rows_given_and_skip_masked_pixels():
    sigma = np.ones((10, N_BANDS))
    sigma[5:] = 100.0
    mask = np.ones_like(sigma, dtype=np.int8)
    mask[:5, 0] = 0
    scales = asinh.fit_band_scales(sigma[:6], mask[:6])
    assert scales[0] == 100.0, "masked pixels must not enter the median"
    np.testing.assert_allclose(scales[1:], 1.0)


def test_band_scales_reject_a_band_with_no_measurement():
    sigma = np.ones((4, N_BANDS))
    mask = np.ones_like(sigma, dtype=np.int8)
    mask[:, 3] = 0
    with pytest.raises(ValueError, match="bands \\[3\\]"):
        asinh.fit_band_scales(sigma, mask)


# --------------------------------------------------------------------------- #
# broad-band RMS normalisation
# --------------------------------------------------------------------------- #
NORM = [BAND_NAMES.index(b) for b in features.NORM_BANDS]


def _photometry(n_rows: int = 8, masked=()):
    rng = np.random.default_rng(0)
    flux = rng.normal(20.0, 10.0, size=(n_rows, N_BANDS))
    sigma = np.abs(flux) * 0.2 + 1.0
    mask = np.ones((n_rows, N_BANDS), dtype=np.int8)
    for row, band in masked:
        mask[row, band] = 0
        sigma[row, band] = np.nan
    return flux, sigma, mask


def test_brightness_is_the_rms_of_the_measured_norm_bands():
    flux, sigma, mask = _photometry(masked=[(1, NORM[0])])
    out = features.normalise(flux, sigma, mask)
    expected0 = np.sqrt(np.mean(flux[0, NORM] ** 2))
    expected1 = np.sqrt(np.mean(flux[1, NORM[1:]] ** 2))
    np.testing.assert_allclose(np.exp(out["logr"][:2]), [expected0, expected1], rtol=1e-5)


def test_every_band_is_divided_by_the_same_brightness():
    flux, sigma, mask = _photometry()
    out = features.normalise(flux, sigma, mask)
    recovered = out["g"].astype(np.float64) * np.exp(out["logr"].astype(np.float64))[:, None]
    np.testing.assert_allclose(recovered, flux, rtol=1e-5)


def test_normalisation_is_scale_free():
    flux, sigma, mask = _photometry()
    base = features.normalise(flux, sigma, mask)
    scaled = features.normalise(1000.0 * flux, 1000.0 * sigma, mask)
    np.testing.assert_allclose(scaled["g"], base["g"], rtol=1e-5)
    np.testing.assert_allclose(scaled["v"], base["v"], rtol=1e-5)
    np.testing.assert_allclose(scaled["sig_logr"], base["sig_logr"], rtol=1e-5)
    np.testing.assert_allclose(scaled["logr"] - base["logr"], np.log(1000.0), rtol=1e-5)


def test_masked_band_gets_zero_shape_and_zero_error():
    flux, sigma, mask = _photometry(masked=[(0, 7)])
    out = features.normalise(flux, sigma, mask)
    assert out["g"][0, 7] == 0.0 and out["v"][0, 7] == 0.0 and out["mask"][0, 7] == 0


def test_shape_error_carries_the_calibration_floor():
    flux = np.full((1, N_BANDS), 50.0)
    sigma = np.zeros_like(flux)
    mask = np.ones_like(flux, dtype=np.int8)
    out = features.normalise(flux, sigma, mask)
    np.testing.assert_allclose(out["v"], features.RELATIVE_CALIB_FLOOR, rtol=1e-5)


def test_brightness_error_is_the_delta_method_on_the_norm_bands():
    flux, sigma, mask = _photometry()
    out = features.normalise(flux, sigma, mask, floor_frac=0.0)
    f, s = flux[:, NORM], sigma[:, NORM]
    r2 = np.mean(f**2, axis=1)
    expected = np.sqrt((f**2 * s**2).sum(axis=1)) / (3 * r2)
    np.testing.assert_allclose(out["sig_logr"], expected, rtol=1e-5)


def test_object_with_every_norm_band_masked_is_rejected():
    flux, sigma, mask = _photometry(masked=[(0, j) for j in NORM])
    with pytest.raises(ValueError, match="every norm band masked"):
        features.normalise(flux, sigma, mask)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
TILES = np.repeat([2241, 2243, 2406, 2470], [50, 40, 30, 20])


def test_each_tile_is_the_test_tile_exactly_once():
    tests = [rotation_tiles(TILES, k)["test"][0] for k in range(4)]
    assert sorted(tests) == [2241, 2243, 2406, 2470]


@pytest.mark.parametrize("k", range(4))
def test_rotation_roles_are_disjoint_and_cover_every_tile(k):
    tiles = rotation_tiles(TILES, k)
    assigned = tiles["train"] + tiles["val"] + tiles["test"]
    assert sorted(assigned) == [2241, 2243, 2406, 2470]
    assert len(tiles["train"]) == 2
    assert tiles["val"] != tiles["test"]


def test_holdout_and_severe_objects_get_no_training_role():
    holdout = np.zeros(len(TILES), dtype=bool)
    holdout[:5] = True
    quality = np.array(["clean"] * len(TILES), dtype=object)
    quality[5:8] = "severe"
    quality[8:10] = "window"
    for k in range(4):
        roles = assign_roles(TILES, holdout, quality, k)
        assert (roles[:5] == "holdout").all()
        assert (roles[5:8] == "excluded").all()
        assert set(roles[8:]) <= {"train", "val", "test"}


def test_rotation_out_of_range_is_rejected():
    with pytest.raises(ValueError, match="rotation must be"):
        rotation_tiles(TILES, 4)


# --------------------------------------------------------------------------- #
# integration against the shipped export
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not RAW_FILE.exists(), reason="raw miniJPAS export not present")
def test_real_export_pipeline_invariants():
    data = load_raw(RAW_FILE)
    assert len(data) == 64_327
    assert len(data.columns) == 137

    deduped = drop_duplicate_sources(data)
    assert len(deduped) == 64_293

    flux, sigma, mask = build_photometry(deduped)
    masked = mask.to_numpy() == 0
    assert 0 < masked.mean() < 0.02, "masked fraction should be small but non-zero"

    meta = clean_metadata(deduped)
    assert (meta["zph"].dropna() > 0).all(), "a sentinel survived in zph"
    assert (meta["e_zph"].dropna() > 0).all(), "a sentinel survived in e_zph"
    # zsp == 0 is legitimate for a STAR (z ~ 0), never for a galaxy or QSO.
    galaxy_like = meta.loc[meta["spCLASSsdss"] != "STAR", "zsp"].dropna()
    assert (galaxy_like > 0).all(), "a zero or negative zsp survived for a galaxy/QSO"
    assert (meta.loc[meta["spCLASSsdss"] == "STAR", "zsp"].dropna() >= 0).all()
    assert meta["zph"].notna().sum() < len(meta) * 0.25
