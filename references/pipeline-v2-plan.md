# Pipeline v2: data processing and model plan

Agreed 2026-11-06 in a design interview. Claude implements, Johan reviews; one
commit per task. Tavla: project `jpda`, goals `data`, `euclidean-vae`,
`eval-harness` (task ids are given per item below).

This plan supersedes `references/vae-normalisation-and-loss.md` where they
disagree: the normaliser (Sec. A4), the justification and form of the additive
floor (Sec. B3), and the shape-only unit-sphere target.

## Decisions at a glance

| Topic | Decision | Rejected alternatives |
|---|---|---|
| Inference holdout | the 1,168 objects with SDSS spectra (`spCLASSsdss` set) | all 19,873 SDSS matches (training would be ~2 mag fainter than inference) |
| Training | unsupervised; label-based inference is a later goal | probes now |
| Splits | rotate over the 4 tiles: 2 train / 1 val / 1 test, 4 rotations | 3/1 without test; random split |
| Quality | drop severe flags from training, keep neighbours/blends; store a `quality` column | strict `flag==0`; keep all |
| Object types | stars, galaxies and QSOs all kept | galaxies only |
| Brightness | `r_bb` = RMS of measured gSDSS, rSDSS, iSDSS; all 60 bands divided by it | narrow-band RMS (current); rSDSS alone; debiased 60-band RMS |
| Extra features | none (photometry only); morphology later as an ablation | CLASS_STAR, flags |
| Asinh | kept only as the input for the PCA baseline, out of `make data` | delete; VAE input option |
| Brightness in model | conditional VAE: `log r_bb` given to encoder and decoder, not predicted | brightness in latent with scale head |
| Noise floor | fixed brightness-relative floor `w² = v² + ε²` (no band known better than `ε·r_bb`) | fitted 5th-percentile floor; no floor |
| Encoder | conv branch on 54 narrow bands → flatten (no pooling); MLP branch on the 6 other bands + `log r_bb`; plain-MLP encoder as ablation | global pooling (current); conv only |
| Latent / β | β = 1 (standard ELBO), latent sizes 4 / 8 / 16 | β = 3, size 6 |
| Evaluation | unsupervised metrics + weighted-PCA and asinh-PCA baselines | label probes now |

## Measurements behind the decisions

From `data/interim/objects.parquet` (64,293 objects after removing duplicates):

- SDSS matches: 19,873 (31%), median r = 22.0 vs 24.0 for unmatched. Only 5% of
  unmatched objects are brighter than r = 22.8. Spectroscopic: 1,168
  (685 GALAXY, 245 STAR, 238 QSO), 973 with usable `zsp`.
- Narrow-band brightness is mostly noise: median `sqrt(Σσ²)/r` over the 54
  narrow bands is 0.69. Broad-band `r_bb` has median S/N 3.7 (5th percentile
  1.4). 603 objects have at least one of g/r/i masked, none have all three.
- Errors do not vanish at low flux: no measured pixel has σ below 0.064× its
  band median, and pixels with near-zero flux keep a median σ of 0.55× it.
  The smallest `v` in the current arrays belongs to an object with
  `logr = 13.85` (median 4.94). The extreme weight range is caused by very
  bright objects, not by a defect in the error model.
- The 0.026 calibration term already caps per-band S/N at ≈ 38. A cap
  proportional to `|g|` therefore adds nothing; the floor has to be relative to
  the object's brightness.
- 23 pixels have |relerr| = 65504 (float16 maximum): capped, so σ is
  underestimated.
- Quality: `flag_rSDSS` ≠ 0 for 6,581 objects (values 1/2/3 = neighbours /
  blended / both; 4 = saturated, 16 = truncated, ...). `mask_flag_rSDSS` ≠ 0 for
  7,562 (values 1, 2, 3; meaning not yet documented). Severe cut as currently
  defined (flag bits ≥ 4, or `mask_flag` ≠ 0) removes 7,616 objects.
- Cross-tile duplicates within 1″: 12 pairs, so the tile split does not leak.
- Training pool after holdout and severe cut: 55,613 objects.

## A. Data processing pipeline

### A1. Inference holdout (`data-labels`)
- `holdout = spCLASSsdss.notna()` (1,168 objects). These rows never enter
  training, validation, testing, or any fitted quantity (ε checks, asinh `b_j`,
  PCA).
- Photometric-only SDSS matches stay in the training pool, without labels.
- Record which labels exist for later inference: `zsp`/`e_zsp`/`f_zsp`,
  `spCLASSsdss`, and for all matches `photoCLASSsdss` and `zph`. Update
  `references/data_dictionary.md`.

### A2. Quality and sample selection (`data-selection`)
- Add a `quality` column in `dataset.py`: `clean` / `blend` (flag bits 1, 2
  only) / `severe` (flag bits ≥ 4, or `mask_flag` ≠ 0).
- **Open, resolve first:** what `mask_flag_rSDSS` values 1/2/3 mean (miniJPAS
  documentation, Bonoli et al. 2021; ask the literature agent). If a value turns
  out to be harmless, it moves from `severe` to `clean`. The 7,616 figure above
  will change accordingly.
- Training uses `clean` + `blend`. Predictions are still written for every
  object, with `quality` attached.
- Set the 23 pixels with capped relerr to `mask = 0`.
- Stars, galaxies and QSOs are all kept.

### A3. Splits (`data-preprocessing`)
- `splits.py` writes one table: `object_id`, `tile_id`, `holdout`, `quality`.
- Rotation k (k = 0..3), with tiles in sorted order `T`: test = `T[k]`,
  validation = `T[(k+1) % 4]`, training = the other two. Derived in code from k,
  so the four rotations cannot drift. Config key `data.rotation`.
- Holdout and `severe` rows are excluded from all three roles.

### A4. Normalisation (`data-preprocessing`)
Moves into `features.py` (CCDS feature step); output stays
`data/processed/vae_inputs.npz`.

```text
r_bb     = sqrt( mean over measured {g,r,i} of f² )
g_j      = f_j / r_bb                                  all 60 bands
v_j      = sqrt( σ_j² + (0.026 f_j)² ) / r_bb          0 where mask_j = 0
logr     = log r_bb
sig_logr = delta-method error of log r_bb from the g/r/i errors
```

- Narrow and broad bands share one scale, so the narrow-vs-broad colour
  information is kept. No projection onto the unit sphere.
- The error of `r_bb` affects every band together. Diagonal approximation:
  add `(g_j · sig_logr)²` to `w²`, behind a config flag (default on). This
  overstates independence, which is documented.
- `r_bb` is still noisy for the faintest objects (5% have S/N < 1.4). No
  debiasing in v2; report the S/N distribution and revisit if reconstruction
  quality depends strongly on it.
- Arrays are stored per block: `narrow` (54), `other` (uJAVA, uJPAS, gSDSS,
  rSDSS, iSDSS, J1007), plus `logr`, `sig_logr`, and the A3 table.

### A5. Asinh representation (`eval-baselines`)
- Removed from `make data`. Built at baseline-fit time only, with `b_j` = median
  σ per band over the **training tiles of the rotation**. This removes the
  current catalogue-wide statistic.

### A6. Module layout and loader (`data-module`)
- `dataset.py`: raw → interim (+ `quality`, capped-relerr mask).
  `band_scales` leaves it.
- `splits.py`: holdout + rotations. `features.py`: RMS normalisation.
  `modeling/data.py`: loader only (`load(rotation) → train/val/test/holdout
  indices + tensors`).
- Tests: normaliser invariants (scale-free, masked bands get zero weight),
  rotation partitions are disjoint and cover the pool, holdout never appears in
  any role, round trip `f = g · r_bb`.

## B. Model implementation

### B1. Encoder (`vae2-encoder`)
- **Narrow branch:** input `(3, 54)` = `g, v, mask` → 2–3 `Conv1d`, k = 5
  (default widths [32, 64]) → **flatten** → linear.
- **Other branch:** 6 bands × `(g, v, mask)` + `log r_bb` (and optionally
  `sig_logr`) → MLP.
- **Fusion:** concatenate → MLP → `(mu, logvar)`.
- **Ablation:** `model.encoder: conv | mlp`. `mlp` flattens all 60 bands ×
  channels + `log r_bb` into one MLP.
- Channel selection stays a config knob (the `[g, mask]` vs `[g, v, mask]`
  ablation tests how much depth leaks into the latent through `v`).
- Encoder and decoder trunks are geometry-agnostic, so the Lorentzian VAE only
  replaces the latent head and the KL.

### B2. Conditional decoder (`vae2-cvae`)
- Input `[z, log r_bb]` → MLP (mirrors the encoder widths) → 60 outputs `ĝ`.
- No unit-sphere projection and no scale head: brightness is a condition, not
  predicted.

### B3. Likelihood (`vae2-cvae`)
- Gaussian with known variance: `w_j² = v_j² + ε² (+ (g_j·sig_logr)²)`.
- `ε` is fixed in the config. Default 0.01, as a uniform floor; per-band ε is
  an option. Before fixing the default, redo the Fisher-spread check (the share
  of total weight held by the top objects) under the `r_bb` normaliser,
  on training tiles only.
- Loss registry: `gaussian_nll` (its `log w²` term is a constant with fixed
  variance; the docstring says so), `student_t` kept, `mse` kept as the
  unweighted contrast case. `weighted_mse` is removed: it has the same
  gradients as `gaussian_nll`.

### B4. Training (`vae2-train`)
- β = 1, linear warm-up kept (config).
- Validation and test use `mu_z` (no sampling). Save the best checkpoint by
  validation loss as well as the last.
- Log KL per latent dimension each epoch.
- `predict.py` writes latents for every object (including holdout), with
  `role`, `quality`, `logr` and per-object χ².

### B5. Latent sweep (`vae2-sweep`)
- Latent size ∈ {4, 8, 16} × 4 rotations = 12 runs. Then the plain-MLP
  encoder and the `[g, mask]` channel set at the chosen size (8 more runs).
- Runs go through the cluster job template; results land in
  `reports/<run>/`.

## C. Evaluation, unsupervised (`eval-metrics`, `eval-baselines`, `eval-report`)
- Metrics on the test tile: reconstruction χ²/dof in `g` space with the same
  `w`, binned by `r_bb` S/N and by tile; KL per dimension and the number of
  active dimensions (KL > 0.01 nats); spread across the 4 rotations.
- Baselines at the same latent size: weighted PCA on `g` (weights `1/w²`), and
  PCA on asinh `x`, whose reconstruction is mapped back to flux and scored with
  the same χ².
- One table per sweep in `reports/`, with every run, every baseline and every
  rotation.

## Order

```text
data-labels ─┐
data-eda ────┴─▶ data-selection ─▶ data-preprocessing ─▶ data-module
                                                           │
                       ┌───────────────────────────────────┤
                       ▼                                   ▼
              vae2-encoder, vae2-cvae              eval-metrics
                       └────────▶ vae2-train ◀──────────┘
                                     │
                       eval-baselines│
                             └──▶ vae2-sweep ─▶ eval-report
```

## Open items

1. Meaning of `mask_flag_rSDSS` values 1/2/3 (A2), which sets the severe cut.
2. ε default after redoing the Fisher-spread check (B5).
3. Whether to debias `r_bb` for faint objects (A4).
4. Conv widths and depth; whether `sig_logr` enters the encoder as a condition.
5. Later goal, out of scope here: label-based inference on the holdout
   (redshift, class, anomalies) and morphology as an extra feature.
