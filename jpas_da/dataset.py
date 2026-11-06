"""Raw miniJPAS export -> interim object table.

This module does the *only* irreversible work in the pipeline: turning the SQL
export into one row per astronomical object with flux, an absolute flux
uncertainty, and an explicit validity mask. It does not choose a model
representation; that is :mod:`jpas_da.features`.

What it fixes, and why (all measured on the shipped export, 64,327 rows):

* ``FLUX_RELERR_AUTO`` is a *signed relative* error, not an absolute one.
  It agrees in sign with the flux for 99.1% of pixels, and ``1/|rel|`` behaves
  as the signal-to-noise ratio (median 1.1 in a narrow band, 4.6 in rSDSS).
  So ``sigma_f = |flux * relerr|``.
* ``relerr == 0`` means the uncertainty was not computed, not that the source
  is infinitely bright: 0.94% of pixels, concentrated on faint pixels of faint
  objects and in the noisiest bands (J0910, J1007, J0900, J0378, uJPAS), and
  16.7k of them in tile 2241 against 3.7k in tile 2406. They become
  ``sigma = NaN`` with ``mask = 0``, so no downstream step can silently treat
  them as high signal-to-noise.
* ``|relerr| == 65504`` is the float16 maximum: a clipped value whose derived
  sigma is an underestimate (23 pixels). Those pixels are masked like the
  ``relerr == 0`` ones.
* Every object gets a ``quality`` class (``clean`` / ``blend`` / ``window`` /
  ``severe``) from the r-band FLAGS and MASK_FLAGS; see :func:`quality_class`.
  Nothing is dropped here: the training selection happens in :mod:`splits`.
* 34 ``(tile_id, number)`` pairs are duplicated. Fluxes are identical within
  each group; the duplication lives in the SDSS cross-match (different
  ``angDist``, sometimes different ``zph``). Keep the smallest-``angDist`` row.
* Sentinels are values, not NaNs. ``zph = -9999`` (813 rows, with
  ``e_zph = -9999`` too) and ``zsp < 0`` (193 rows, 192 of them STARs at z~0)
  become NaN. ``zsp == 0`` is kept for the four STARs, where it is a measured
  value, and dropped for the two GALAXY rows that also carry ``e_zsp = 0`` and
  ``f_zsp = 646`` while their photo-z is ~0.5. ``e_zsp == 0`` is also
  dropped (3 rows): a zero uncertainty would read as infinite weight.
* ``MAGrSDSS`` is exactly ``26.1147 - 2.5*log10(rSDSS)`` (residual std 1e-6)
  and ``angDist`` is non-null for exactly the SDSS-cross-matched rows; either
  would leak label availability into a model. They stay as metadata, never as
  flux. The ``26.1147`` zeropoint is also the flux unit of the whole export:
  these are not uJy.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from jpas_da.bands import BAND_NAMES, ERROR_COLUMN
from jpas_da.config import INTERIM_DATA_DIR, RAW_DATA_DIR

app = typer.Typer()

RAW_FILE = RAW_DATA_DIR / "miniJpas_fluxes_errors_redshifts_full.csv"
OBJECTS_FILE = INTERIM_DATA_DIR / "objects.parquet"
BAND_SCALES_FILE = INTERIM_DATA_DIR / "band_scales.parquet"

# Non-photometric columns. Kept in the interim table as metadata so that later
# inference work never has to re-read the 75 MB CSV, and so that any accidental
# use of them as features is visible in a diff rather than hidden in a join.
METADATA_COLUMNS = (
    "tile_id",
    "number",
    "ALPHA_J2000",
    "DELTA_J2000",
    "MAGrSDSS",
    "flag_rSDSS",
    "mask_flag_rSDSS",
    "SEtractor",
    "photoCLASSsdss",
    "spCLASSsdss",
    "zsp",
    "e_zsp",
    "f_zsp",
    "zph",
    "e_zph",
    "avg_zph",
    "angDist",
)

# Numeric sentinels meaning "no value". zsp/e_zsp/f_zsp/zph/e_zph all use
# negative values (zph and e_zph use exactly -9999); MAGrSDSS uses 99.
SENTINEL_MAX = 0.0
MAG_SENTINEL_MIN = 30.0

# The relative errors are stored as float16 in the archive: 65504 is the
# largest finite float16, so |relerr| == 65504 is a clipped value and the
# absolute error derived from it is an underestimate (23 pixels).
RELERR_CAP = 65504.0

# SExtractor FLAGS bits 1 (neighbours/bad pixels) and 2 (blended) are common and
# leave usable photometry; any higher bit (4 saturated, 8 truncated, 16 corrupted
# aperture, ...) does not. miniJPAS MASK_FLAGS (Bonoli et al. 2021, Table 6):
# 1 = outside the window frame (less exposure), 2 = bright star or near one,
# 4 = artefact.
FLAGS_BENIGN = 1 | 2
MASK_WINDOW = 1
MASK_SEVERE = 2 | 4
QUALITY_CLASSES = ("clean", "blend", "window", "severe")
TRAINING_QUALITY = ("clean", "blend", "window")


def load_raw(path: Path) -> pd.DataFrame:
    """Read the raw CSV export.

    The file starts with the SQL query used to produce it (a single,
    comma-separated line). Skip that line when it is present.
    """
    with path.open() as f:
        first_line = f.readline()

    skiprows = 1 if first_line.lstrip().startswith("#SELECT") else 0
    data = pd.read_csv(path, skiprows=skiprows)
    data.columns = data.columns.str.strip()
    return data


def validate_columns(data: pd.DataFrame) -> None:
    """Fail loudly if the export is not the one this module was written for."""
    needed = [*BAND_NAMES, *ERROR_COLUMN.values(), *METADATA_COLUMNS]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ValueError(f"raw export is missing {len(missing)} expected columns: {missing}")
    duplicated = data.columns[data.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"duplicated column names in raw export: {duplicated}")


def drop_duplicate_sources(data: pd.DataFrame) -> pd.DataFrame:
    """Collapse duplicated ``(tile_id, number)`` pairs onto the best match.

    Within a duplicated group the fluxes are identical and ``angDist`` differs,
    so the smallest ``angDist`` is the tightest SDSS correspondence and the one
    whose labels are most trustworthy.
    """
    ordered = data.sort_values(["tile_id", "number", "angDist"], na_position="last")
    deduped = ordered.drop_duplicates(["tile_id", "number"], keep="first")
    n_dropped = len(data) - len(deduped)
    if n_dropped:
        logger.info(f"Dropped {n_dropped} duplicate (tile_id, number) rows")
    return deduped.sort_values(["tile_id", "number"]).reset_index(drop=True)


def clean_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Replace numeric sentinels by NaN so they cannot be mistaken for values.

    Three sentinel rules, each with its own justification:

    * negative ``zsp``/``e_zsp``/``f_zsp``/``zph``/``e_zph`` -> NaN (193 zsp
      rows, 813 zph rows).
    * ``zsp == 0`` is kept only for STARs. Six rows hold exactly zero: four
      STARs, where z = 0 is the measured value, and two GALAXY rows whose
      ``e_zsp`` is also 0 and whose ``f_zsp`` is 646 while their photo-z is
      ~0.5. Those two are unusable and would otherwise survive a naive
      ``zsp > 0`` selection as z = 0 galaxies.
    * ``e_zsp == 0`` / ``e_zph == 0`` means the uncertainty was not computed,
      not that it is zero: such an error would be treated as infinite weight.
      Three ``e_zsp`` values are affected; none of ``e_zph``.
    * ``MAGrSDSS`` uses 99 as "not measured"; that is also the value stored for
      913 rows.
    """
    meta = data[list(METADATA_COLUMNS)].copy()
    for column in ("zsp", "e_zsp", "f_zsp", "zph", "e_zph"):
        values = np.asarray(meta[column], dtype=float)
        values[values < SENTINEL_MAX] = np.nan
        meta[column] = values

    for column in ("e_zsp", "e_zph"):
        values = np.asarray(meta[column], dtype=float)
        values[values == 0.0] = np.nan
        meta[column] = values

    spectroscopic = np.asarray(meta["zsp"], dtype=float)
    is_star = np.asarray(meta["spCLASSsdss"], dtype=object) == "STAR"
    spectroscopic[(spectroscopic == 0.0) & ~is_star] = np.nan
    meta["zsp"] = spectroscopic

    magnitude = np.asarray(meta["MAGrSDSS"], dtype=float)
    magnitude[~np.isfinite(magnitude) | (magnitude > MAG_SENTINEL_MIN)] = np.nan
    meta["MAGrSDSS"] = magnitude
    return meta


def label_flags(meta: pd.DataFrame) -> pd.DataFrame:
    """Which objects form the inference holdout, and which redshifts to trust.

    * ``holdout``: the object has an SDSS spectrum (``spCLASSsdss`` set; 1,168
      objects). These are kept for label-based inference and never enter
      training, validation, testing or any fitted quantity. Photometric-only
      SDSS matches are *not* held out: the model is unsupervised and never reads
      their labels, and holding them out would leave training ~2 mag fainter
      than the inference set.
    * ``zsp_reliable``: a spectroscopic redshift is present and ``f_zsp`` (the
      SDSS ``ZWARNING`` bitmask) is 0. 250 holdout objects carry
      ``ZWARNING = 4`` (small delta chi2, i.e. an ambiguous fit) and a handful
      carry other bits; their redshifts are kept but not marked reliable.
    """
    holdout = meta["spCLASSsdss"].notna().to_numpy()
    zsp_reliable = (meta["zsp"].notna() & (meta["f_zsp"] == 0)).to_numpy()
    return pd.DataFrame({"holdout": holdout, "zsp_reliable": zsp_reliable & holdout})


def quality_class(flags: np.ndarray, mask_flags: np.ndarray) -> np.ndarray:
    """One quality label per object from the r-band FLAGS and MASK_FLAGS.

    First match wins:

    * ``severe``: a FLAGS bit above 2 (saturated, truncated, corrupted aperture,
      ...) or a MASK_FLAGS bright-star/artefact bit. The photometry itself is
      compromised; excluded from training.
    * ``window``: outside the window frame only. The object got less exposure,
      which its quoted errors already carry, so it is kept; evaluation reports it
      separately.
    * ``blend``: neighbours or blended (FLAGS bits 1, 2 only).
    * ``clean``: nothing set.
    """
    flags = np.asarray(flags).astype(np.int64)
    mask_flags = np.asarray(mask_flags).astype(np.int64)
    quality = np.full(len(flags), "clean", dtype=object)
    quality[(flags & FLAGS_BENIGN) != 0] = "blend"
    quality[(mask_flags & MASK_WINDOW) != 0] = "window"
    severe = ((flags & ~FLAGS_BENIGN) != 0) | ((mask_flags & MASK_SEVERE) != 0)
    quality[severe] = "severe"
    return quality


def build_photometry(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the photometry into flux, absolute sigma and a validity mask.

    Returns ``(flux, sigma, mask)``, each with one column per band, named after
    the band and prefixed ``flux_``, ``sigma_`` or ``mask_``. ``sigma`` is NaN
    exactly where ``mask`` is 0, so the two cannot disagree.
    """
    flux = data[list(BAND_NAMES)].to_numpy(dtype=float)
    relerr = data[[ERROR_COLUMN[b] for b in BAND_NAMES]].to_numpy(dtype=float)
    defined = (relerr != 0.0) & (np.abs(relerr) < RELERR_CAP)
    sigma = np.abs(flux * relerr)
    sigma[~defined] = np.nan

    def frame(values: np.ndarray, prefix: str, dtype=None) -> pd.DataFrame:
        return pd.DataFrame(values, columns=[f"{prefix}{b}" for b in BAND_NAMES], dtype=dtype)

    n_pixels = flux.size
    logger.info(
        f"{int((~defined).sum()):,}/{n_pixels:,} pixels "
        f"({100 * (~defined).mean():.2f}%) have no usable relative error -> masked "
        f"({int((np.abs(relerr) >= RELERR_CAP).sum())} of them clipped at {RELERR_CAP:g})"
    )
    logger.info(
        f"{int((~defined).any(axis=1).sum()):,}/{len(flux):,} objects have >=1 masked band; "
        f"{int(defined.all(axis=1).sum()):,} are complete"
    )
    return frame(flux, "flux_"), frame(sigma, "sigma_"), frame(defined.astype("int8"), "mask_")


def band_scales(objects: pd.DataFrame) -> pd.DataFrame:
    """Per-band soft scale ``b_j = median |sigma_f|`` over the whole sample.

    ``b_j`` is the pivot of the asinh representation (see
    :mod:`jpas_da.features`): it separates the noise-dominated part of each
    band (linear regime) from the source-dominated part (log regime).

    It is computed globally, not per tile, and that is deliberate. Median
    log-sigma differs between tiles by 0.16 dex typically and up to 0.37 dex in
    the worst bands, so a held-out tile does sit off-centre in ``x``. That
    offset is a real property of the survey depth and leave-one-tile-out is
    supposed to expose it; hiding it inside the normalisation would make the
    held-out field look easier than it is.
    """
    sigma = objects[[f"sigma_{b}" for b in BAND_NAMES]].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        finite = np.isfinite(sigma)
        positive = sigma > 0
        median_sigma = np.array(
            [np.median(sigma[:, j][finite[:, j] & positive[:, j]]) for j in range(sigma.shape[1])]
        )
    n_defined = finite.sum(axis=0)
    scale = pd.DataFrame(
        {
            "band": list(BAND_NAMES),
            "b_j": median_sigma,
            "n_sigma_defined": n_defined,
            "n_sigma_used": (finite & positive).sum(axis=0),
        }
    )
    if not np.isfinite(scale["b_j"]).all() or (scale["b_j"] <= 0).any():
        bad = scale.loc[~np.isfinite(scale["b_j"]) | (scale["b_j"] <= 0), "band"].tolist()
        raise ValueError(f"could not derive a positive b_j for bands: {bad}")
    return scale


def build_interim(raw_path: Path = RAW_FILE, output_path: Path = OBJECTS_FILE) -> pd.DataFrame:
    data = load_raw(raw_path)
    validate_columns(data)
    logger.info(f"Loaded {len(data):,} rows and {data.shape[1]} columns from {raw_path}")
    data = drop_duplicate_sources(data)

    flux, sigma, mask = build_photometry(data)
    meta = clean_metadata(data)
    object_id = (
        meta["tile_id"].astype("int64").astype(str)
        + "-"
        + meta["number"].astype("int64").astype(str)
    )

    labels = label_flags(meta)
    labels["quality"] = quality_class(meta["flag_rSDSS"], meta["mask_flag_rSDSS"])
    counts = labels["quality"].value_counts()
    logger.info("quality: " + ", ".join(f"{q} {int(counts.get(q, 0)):,}" for q in QUALITY_CLASSES))
    logger.info(
        f"inference holdout (SDSS spectra): {int(labels['holdout'].sum()):,} objects, "
        f"{int(labels['zsp_reliable'].sum()):,} with a reliable zsp"
    )

    objects = pd.concat(
        [
            pd.DataFrame({"object_id": object_id, "tile_id": meta["tile_id"].astype("int64")}),
            meta.drop(columns="tile_id"),
            labels,
            flux,
        ],
        axis=1,
    )
    objects = pd.concat([objects, sigma, mask], axis=1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    objects.to_parquet(output_path, index=False)
    logger.success(f"Wrote {output_path} ({len(objects):,} objects, {objects.shape[1]} columns)")

    scales = band_scales(objects)
    scales_path = output_path.with_name(BAND_SCALES_FILE.name)
    scales.to_parquet(scales_path, index=False)
    logger.success(f"Wrote {scales_path}")

    logger.info(
        "zph usable: {:,} | zsp usable: {:,} | spCLASS: {:,}".format(
            int(objects["zph"].notna().sum()),
            int(objects["zsp"].notna().sum()),
            int(objects["spCLASSsdss"].notna().sum()),
        )
    )
    return objects


@app.command()
def main(
    input_path: Path = RAW_FILE,
    output_path: Path = OBJECTS_FILE,
):
    build_interim(input_path, output_path)


if __name__ == "__main__":
    app()
