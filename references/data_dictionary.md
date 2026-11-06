# Data dictionary

What the pipeline produces, and what every number in it means. All figures are
measured on the shipped export (`data/raw/miniJpas_fluxes_errors_redshifts_full.csv`,
64,327 rows x 137 columns) and are reproducible with `make data features splits`.

## Pipeline

| Stage | Command | Output |
|---|---|---|
| 1 | `make data` | `data/interim/objects.parquet`, `data/interim/band_scales.parquet` |
| 2 | `make features` | `data/processed/jpas_asinh.parquet` |
| 3 | `make splits` | `data/interim/folds.parquet` |

`make datasets` runs stages 2 and 3. Each stage is idempotent and reads only the
stage before it, so the representation in stage 2 can be replaced without
touching the raw export.

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

## Band scales (`data/interim/band_scales.parquet`)

One row per band: `band`, `b_j`, `n_sigma_defined`, `n_sigma_used`.

`b_j = median |sigma_f|` over the whole sample - the pivot of the asinh
transform, i.e. the flux at which a band switches from noise-dominated to
source-dominated. Global rather than per tile: median log-sigma differs between
tiles by 0.16 dex typically and up to 0.37 dex in the worst bands, so a
held-out tile sits off-centre in `x`. That offset is a real property of the
survey depth and leave-one-tile-out exists to expose it.

## Representation (`data/processed/jpas_asinh.parquet`)

182 columns: `object_id`, `tile_id`, plus `x_<band>`, `sigma_x_<band>`,
`mask_<band>`.

```
x_b      = asinh(f_b / b_b) * 2
sigma_x  = sigma_f / sqrt(f_b^2 + b_b^2) * 2
f_b      = b_b * sinh(x_b / 2)            # exact inverse
```

`2` is `features.PRESENTATION_SCALE`: a monotonic global factor that only fixes
the numerical size of the channels. Round-trip is asserted at build time.

### Why this representation

| Transform | skew | kurtosis | sigma_out p50 | sigma_out over flux deciles | undefined where |
|---|---|---|---|---|---|
| raw flux | 411 | 202,626 | 9.65 | x4.8 | - |
| `asinh(f/1)` | -0.51 | 0.16 | 0.69 | x36 | - |
| `asinh(f/b_j)` | 1.35 | 4.6 | 0.46 | x5.8 | - |
| `log10 f` (f>0) | 0.73 | 3.72 | 0.29 | x84 | 21.7% of pixels |

- **Asinh handles the data as measured.** 36% of measured pixels have |SNR|<1 and
  21.7% have negative flux, so `log` needs an arbitrary floor and `f/sigma`
  encodes SNR instead of flux.
- **The output noise is flat.** With `b_b = median sigma`, `sigma_x` spans only
  0.53..0.43 over the nine faintest flux deciles and drops to 0.10 on the
  brightest, where the measurement really is better. corr(sigma_x, |f|) = -0.01.
- **The geometry is sane.** In raw flux the brightest 0.1% of pixels carry 99.9%
  of `sum(f^2)`; only 60% of the 50 nearest neighbours survive the switch to
  asinh, which is the point.
- **Both physical axes survive.** Per-object brightness is 69.5% of the variance
  (65% mean level, 35% SED shape); brightness is kept in the channels rather
  than normalised away.

Masked pixels get `x = 0` and `sigma_x = NaN`. Zero is unambiguous as "no
measurement": real non-detections scatter around `|x| ~ 1`.

## Folds (`data/interim/folds.parquet`)

`object_id`, `tile_id`, `fold`. Leave-one-tile-out: fold *k* holds out one
pointing, so every object of that pointing is in exactly one test set.

| Fold | Held-out tile | Test objects | Train objects |
|---|---|---|---|
| 0 | 2241 | 20,016 | 44,277 |
| 1 | 2243 | 13,836 | 50,457 |
| 2 | 2406 | 15,792 | 48,501 |
| 3 | 2470 | 14,649 | 49,644 |

Use `jpas_da.features.load_arrays()` to get `x`, `sigma_x`, `mask`,
`object_id`, `tile_id` and `fold` as aligned arrays.

## Walkthrough

`notebooks/explore_data.py` (marimo) shows all of the above on the real files:
shapes and load pattern, the 60-channel layout, the histogram of `x` against
raw flux, `b_j`, the verification of the inverse, the flatness of `sigma_x`,
one object plotted in both coordinates, the mask distribution, the
brightness/shape variance split, and the folds.

```bash
marimo edit notebooks/explore_data.py
```
