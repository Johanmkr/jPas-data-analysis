# Data dictionary

What the pipeline produces, and what every number in it means. All figures are
measured on the shipped export (`data/raw/miniJpas_fluxes_errors_redshifts_full.csv`,
64,327 rows x 137 columns) and are reproducible with `make data`.

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1 | `make interim` | `data/interim/objects.parquet` (flux, sigma, mask, labels, `holdout`, `quality`) |
| 2 | `make splits` | `data/interim/splits.parquet` (roles per tile rotation) |
| 3 | `make features` | `data/processed/vae_inputs.npz` (broad-band RMS normalisation) |

`make data` runs all three. Each stage is idempotent and reads only the
interim table, so the normalisation can be replaced without touching the raw
export. The asinh representation is no longer a pipeline stage: it is built
inside the PCA baseline from the training tiles of a rotation
(`jpas_da/asinh.py`).

## What one row is

One astronomical object: a unique `(tile_id, number)` in the miniJPAS
`FLambdaDualObj` catalogue. 64,293 objects after removing 34 duplicated pairs.

| Property | Value |
|---|---|
| Tiles | 2241, 2243, 2406, 2470 (four overlapping pointings) |
| Objects | 64,293 |
| Bands | 60 |
| Objects with all 60 uncertainties measured | 60,991 |
| Pixels with no measured uncertainty | 38,916 (1.01%) |
| `zph` usable | 12,419 |
| `zsp` usable | 973 |
| `spCLASSsdss` present | 1,168 |

Objects are never shared between tiles; neighbouring objects *within* a tile
share sky background and per-pointing calibration, which is why splits are
grouped by tile.

## Bands

Order and wavelengths live in `jpas_da/bands.py`; 60 channels in four blocks:

| Block | n | Members | Notes |
|---|---|---|---|
| narrow | 54 | `J0378` ... `J0910` | 10 A grid, except `J0378`->`J0390` which is 12 A |
| medium | 1 | `uJAVA` | FWHM 495 A, centred 3497 A |
| broad | 4 | `uJPAS`, `gSDSS`, `rSDSS`, `iSDSS` | `rSDSS` is the detection band |
| highpass | 1 | `J1007` | centred 9316 A |

## Interim table (`data/interim/objects.parquet`)

198 columns: 2 identifiers, 16 metadata, and 3 x 60 photometry columns.

### Identifiers and metadata

`object_id` is `"<tile_id>-<number>"`, e.g. `2241-2`.

| Column | Meaning | Notes |
|---|---|---|
| `tile_id` | miniJPAS pointing | 2241, 2243, 2406, 2470 |
| `number` | object number within the tile | |
| `ALPHA_J2000`, `DELTA_J2000` | sky position (deg) | |
| `MAGrSDSS` | SDSS r magnitude | **leak**: exactly `26.1147 - 2.5*log10(rSDSS)`; 99 -> NaN |
| `flag_rSDSS` | SExtractor `FLAGS` in r, sum of bits | 1 neighbours or bad pixels (4,027), 2 blended (5,166), 4 saturated (59), 8 truncated at image edge (20), 16 incomplete/corrupted aperture data (219), 32/64/128 not present |
| `mask_flag_rSDSS` | miniJPAS `MASK_FLAGS` in r, sum of bits (Bonoli et al. 2021, Table 6) | 0 not masked (56,731), 1 outside the window frame: normalised weight map < 85%, less exposure (6,034 with the bit), 2 bright star or near one (3,445), 4 artefact (not present). Values seen: 1 (4,117), 2 (1,528), 3 (1,917) |
| `SEtractor` | SExtractor `CLASS_STAR` | consistently separated from `photoCLASSsdss` (cross-tab in the profiling run) |
| `photoCLASSsdss` | SDSS photo class | 3 = galaxy, 6 = point source, NaN = not matched |
| `spCLASSsdss` | SDSS spectral class | GALAXY / STAR / QSO |
| `zsp`, `e_zsp`, `f_zsp` | SDSS spec-z, its error, flags | sentinels -> NaN |
| `zph`, `e_zph` | SDSS photo-z and error | `-9999` sentinel -> NaN on 813 rows |
| `avg_zph` | averaged photo-z | always positive where present; corr 0.70 with `zph` |
| `angDist` | angular distance to the SDSS match (arcsec) | **leak**: non-null exactly on cross-matched rows |

The two leak columns are kept deliberately, so that any use of them shows up as
an explicit read in review rather than as an invisible join.

### Photometry

Three blocks of 60 columns, all in canonical band order:

| Prefix | dtype | Meaning |
|---|---|---|
| `flux_<band>` | float64 | `FLUX_AUTO` in the export's flux unit (zeropoint 26.1147; **not** uJy) |
| `sigma_<band>` | float64 | absolute 1-sigma: `abs(flux * FLUX_RELERR_AUTO)`; NaN where unmeasured |
| `mask_<band>` | int8 | 1 if the uncertainty was measured, 0 otherwise |

`FLUX_RELERR_AUTO` is a *signed relative* error: it agrees in sign with the flux
for 99.1% of pixels and `1/|rel|` behaves as SNR (median 1.1 in `J0450`, 4.6 in
`rSDSS`).

**`relerr == 0` means missing, not saturated.** 0.94% of pixels, on faint
pixels of faint objects (72% below their object's median `|flux|`), concentrated
in the noisiest bands (`J0910`, `J1007`, `J0900`, `J0378`, `uJPAS`) and in tile
2241 (16.7k) versus 2406 (3.7k). Treating them as infinite SNR would be wrong,
so they get `sigma = NaN` and `mask = 0`.

## Labels and the inference holdout

The VAE is trained without labels. Labels exist only for SDSS-matched objects
and are used later, for inference on the holdout.

| Label | Present for | Reliability |
|---|---|---|
| `spCLASSsdss` | 1,168 (685 GALAXY, 245 STAR, 238 QSO) | spectroscopic class; defines the holdout |
| `zsp` | 973, all inside the holdout | `f_zsp` is the SDSS `ZWARNING` bitmask: 0 = clean (771: 568 GALAXY, 162 QSO, 41 STAR), 4 = small delta chi2, an ambiguous fit (250 holdout objects, 199 of them with a `zsp`). 192 STARs have no `zsp` (negative sentinel). `e_zsp` p99 is 0.7, set by the flagged fits |
| `photoCLASSsdss` | all 19,873 matches | SDSS morphology, 3 = extended, 6 = point source. Against the spectra: 605/685 GALAXY extended, 232/245 STAR point-like, QSO split 79/159. It is morphology, not class |
| `zph`, `e_zph` | 12,419 matches | SDSS photo-z. For the 584 holdout galaxies with both: NMAD 0.058, 18.5% outliers (\|dz\|/(1+z) > 0.15). Too noisy to be a target |
| `avg_zph` | 13,222 (the extended matches) | averaged photo-z, corr 0.70 with `zph` |

Columns added by `dataset.py`:

- `holdout`: `spCLASSsdss` is set. These 1,168 objects (264-319 per tile;
  median r = 21.5, 5-95% range 17.2-22.7) never enter training, validation,
  testing or any fitted quantity. Photometric-only matches stay in the
  training pool, unlabelled: removing them would leave training ~2 mag fainter
  than the inference set (median r 24.0 against 22.0 for all matches).
- `zsp_reliable`: holdout object with a `zsp` and `f_zsp == 0`.

## Quality classes

`dataset.py` adds `quality` from the two flag columns (first match wins):

| Class | Rule | Objects | Training |
|---|---|---|---|
| `severe` | FLAGS bit >= 4, or MASK_FLAGS bit 2 (bright star) or 4 (artefact) | 3,672 | excluded |
| `window` | MASK_FLAGS bit 1 only: outside the window frame, less exposure | 3,944 | kept, reported separately |
| `blend` | FLAGS bits 1 or 2 only: neighbours, blended | 4,112 | kept |
| `clean` | nothing set | 52,565 | kept |

The 23 pixels whose relative error is clipped at 65504 (the float16 maximum)
are masked like the `relerr == 0` ones.


## Model inputs (`data/processed/vae_inputs.npz`)

Per object, every band is divided by the broad-band brightness:

```text
r_bb     = sqrt( mean over measured {gSDSS, rSDSS, iSDSS} of f^2 )
g_j      = f_j / r_bb                                  all 60 bands
v_j      = sqrt( sigma_j^2 + (0.026 f_j)^2 ) / r_bb    0 where mask_j = 0
logr     = log r_bb
sig_logr = sqrt( sum_k m_k f_k^2 sigma_k^2 ) / (n r_bb^2)
```

| Key | Shape | Meaning |
|---|---|---|
| `g`, `v`, `mask` | (N, 60) | shape, its error, validity; canonical band order (54 narrow, then uJAVA, uJPAS, gSDSS, rSDSS, iSDSS, J1007) |
| `logr`, `sig_logr` | (N,) | log brightness and its error |
| `object_id`, `tile_id`, `holdout`, `quality` | (N,) | identifiers for the splits |
| `n_bands`, `relative_calib_floor` | scalar | 60 and 0.026 |

`log r_bb` has median S/N 6.4 (`1/sig_logr`, 5th percentile 2.5). A
narrow-band brightness would be mostly noise: over the 54 narrow bands, noise
makes up a median 69% of the RMS. Reasons and alternatives:
`references/pipeline-v2-plan.md`, Sec. A4.

## Splits (`data/interim/splits.parquet`)

`object_id`, `tile_id`, `holdout`, `quality`, and `role_0` ... `role_3`. With
tiles sorted, rotation *k* tests on tile *k*, validates on tile *k+1* (mod 4)
and trains on the other two. Holdout objects get role `holdout`, `severe`
objects `excluded`. Training recomputes the roles with
`jpas_da.splits.assign_roles`, so the stored columns are a convenience only.

| Rotation | Test | Val | Train tiles | Train | Val | Test |
|---|---|---|---|---|---|---|
| 0 | 2241 | 2243 | 2406, 2470 | 28,156 | 13,190 | 18,140 |
| 1 | 2243 | 2406 | 2241, 2470 | 31,753 | 14,543 | 13,190 |
| 2 | 2406 | 2470 | 2241, 2243 | 31,330 | 13,613 | 14,543 |
| 3 | 2470 | 2241 | 2243, 2406 | 27,733 | 18,140 | 13,613 |

In every rotation, 1,168 objects are `holdout` and 3,639 `excluded`.

## Walkthrough

`notebooks/explore_data.py` (marimo) still describes the v1 pipeline (asinh
representation, leave-one-tile-out folds); the v2 refresh is tracked in the tavla
task `data-spectra`. It shows, on the real files:
shapes and load pattern, the 60-channel layout, the histogram of `x` against
raw flux, `b_j`, the verification of the inverse, the flatness of `sigma_x`,
one object plotted in both coordinates, the mask distribution, the
brightness/shape variance split, and the folds.

```bash
marimo edit notebooks/explore_data.py
```
