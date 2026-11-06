"""Invariants of the raw -> interim -> representation pipeline.

These are cheap and run offline on small synthetic frames, plus one integration
test that exercises the real export behind a skip when it is absent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from jpas_da import features
from jpas_da.bands import BAND_NAMES, BLOCK_ORDER, N_BANDS, select
from jpas_da.config import RAW_DATA_DIR
from jpas_da.dataset import (
    ERROR_COLUMN,
    METADATA_COLUMNS,
    band_scales,
    build_photometry,
    clean_metadata,
    drop_duplicate_sources,
    label_flags,
    load_raw,
)
from jpas_da.splits import split_by_tile

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
# representation
# --------------------------------------------------------------------------- #
def _objects_from_raw(data: pd.DataFrame) -> pd.DataFrame:
    flux, sigma, mask = build_photometry(data)
    meta = clean_metadata(data)
    identifiers = pd.DataFrame(
        {
            "object_id": meta["tile_id"].astype(str) + "-" + meta["number"].astype(str),
            "tile_id": meta["tile_id"],
        }
    )
    return pd.concat([identifiers, meta.drop(columns="tile_id"), flux, sigma, mask], axis=1)


def test_asinh_roundtrip_recovers_every_defined_flux():
    objects = _objects_from_raw(_synthetic_raw())
    scales = band_scales(objects)
    representation = features.build_representation(objects, scales)
    features.check_roundtrip(representation, objects, scales)

    flux = objects[[f"flux_{b}" for b in BAND_NAMES]].to_numpy()
    x = representation[[f"x_{b}" for b in BAND_NAMES]].to_numpy() / features.PRESENTATION_SCALE
    recovered = features.asinh_inverse(x, scales["b_j"].to_numpy())
    np.testing.assert_allclose(recovered, flux, rtol=1e-10, atol=1e-10)


def test_asinh_is_linear_in_the_noise_and_logarithmic_in_the_signal():
    scale = np.full(N_BANDS, 10.0)
    faint = np.array([[0.5] * N_BANDS])
    bright = np.array([[1000.0] * N_BANDS])
    np.testing.assert_allclose(
        features.asinh_transform(faint, scale), faint / scale, rtol=1e-3
    )
    np.testing.assert_allclose(
        features.asinh_transform(bright, scale), np.log(2 * bright / scale), rtol=1e-5
    )


def test_negative_and_zero_flux_are_representable_and_finite():
    scale = np.full(N_BANDS, 10.0)
    flux = np.array([[-50.0, 0.0, 7.0] * 20])
    x = features.asinh_transform(flux, scale)
    assert np.isfinite(x).all()
    np.testing.assert_allclose(features.asinh_inverse(x, scale), flux, rtol=1e-12)


def test_masked_pixels_get_zero_and_nan_sigma():
    data = _synthetic_raw(undefined_pixel=(2, 7))
    objects = _objects_from_raw(data)
    scales = band_scales(objects)
    representation = features.build_representation(objects, scales)
    band = BAND_NAMES[7]
    assert representation[f"x_{band}"].to_numpy()[2] == 0.0
    assert np.isnan(representation[f"sigma_x_{band}"].to_numpy()[2])
    assert representation[f"mask_{band}"].to_numpy()[2] == 0


def _measured_noise_model(flux: np.ndarray) -> np.ndarray:
    """``sigma_f ~ 0.4|f| + floor``: the shape measured in the export."""
    return 0.4 * np.abs(flux) + 1.0


def test_propagated_error_is_flat_once_the_source_dominates_the_noise():
    """The reason the soft scale is the median sigma, not a flux floor.

    With ``sigma_f = k|f| + floor``, asinh makes ``sigma_x -> k`` (a constant)
    in the source-dominated regime instead of growing with flux as it does in
    raw flux units.
    """
    scale = np.full(N_BANDS, 10.0)
    flux = np.array([[100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0] * 10])
    errors = features.asinh_error(flux, _measured_noise_model(flux), scale)
    assert errors.max() / errors.min() < 1.05, f"sigma_x varies by {errors.max() / errors.min():.2f}x"
    assert np.allclose(errors, 0.4, rtol=0.05), "sigma_x should converge to the relative error"


def test_the_same_noise_model_is_strongly_flux_dependent_in_flux_units():
    """Counter-check for the previous test."""
    flux = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0, 30000.0])
    sigma = _measured_noise_model(flux)
    assert sigma.max() / sigma.min() > 100


def test_brightness_and_shape_both_survive_the_transform():
    """With a common level times a common SED, both axes must remain visible."""
    rng = np.random.default_rng(3)
    sed = np.exp(-0.5 * ((np.arange(N_BANDS) - 25) / 12) ** 2) + 0.2
    level = rng.uniform(30.0, 3000.0, size=(300, 1))
    flux = level * sed * np.exp(rng.normal(0, 0.1, size=(300, N_BANDS)))
    scale = np.full(N_BANDS, float(np.median(_measured_noise_model(flux))))

    x = features.asinh_transform(flux, scale)
    level_share = np.var(x.mean(axis=1)) / np.var(x)
    assert 0.5 < level_share < 0.98, f"level share {level_share:.2f} outside the useful range"

    flat = features.asinh_transform(level * sed, scale)
    flat_share = np.var(flat.mean(axis=1)) / np.var(flat)
    assert flat_share > level_share, "noise should add shape variance, not remove it"


def test_band_scales_are_positive_and_follow_band_order():
    scales = band_scales(_objects_from_raw(_synthetic_raw(n_rows=30)))
    assert list(scales["band"]) == list(BAND_NAMES)
    assert (scales["b_j"] > 0).all()


def test_representation_rejects_misordered_scales():
    objects = _objects_from_raw(_synthetic_raw())
    scales = band_scales(objects).sample(frac=1.0, random_state=0).reset_index(drop=True)
    with pytest.raises(ValueError, match="canonical band order"):
        features.build_representation(objects, scales)


# --------------------------------------------------------------------------- #
# splits
# --------------------------------------------------------------------------- #
def test_split_by_tile_never_shares_a_tile_between_train_and_test():
    tiles = np.repeat([2241, 2243, 2406, 2470], [50, 40, 30, 20])
    folds, splits = split_by_tile(tiles)
    assert len(splits) == 4
    for train, test in splits:
        assert set(tiles[train]).isdisjoint(tiles[test])
        assert len(train) + len(test) == len(tiles)
    assert (folds >= 0).all()


def test_split_by_tile_rejects_a_single_tile():
    with pytest.raises(ValueError, match="at least two tiles"):
        split_by_tile(np.array([2241, 2241, 2241]))


def test_split_by_tile_honours_n_splits():
    tiles = np.repeat([2241, 2243, 2406, 2470], 10)
    folds, splits = split_by_tile(tiles, n_splits=2)
    assert len(splits) == 2
    assert folds.max() == 1


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
