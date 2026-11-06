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
| Quality | drop severe flags (FLAGS ≥ 4, bright-star mask) from training; keep blends and window-frame objects; store a `quality` column | strict `flag==0`; keep all |
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
  narrow bands is 0.69. Broad-band `log r_bb` has median S/N 6.4
  (`1/sig_logr`, 5th percentile 2.5; the per-band average was 3.7). 603 objects have at least one of g/r/i masked, none have all three.
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
- Quality (bit meanings in `references/data_dictionary.md`): `flag_rSDSS`
  (SExtractor FLAGS) has bit 1 (neighbours/bad pixels) on 4,027 objects and
  bit 2 (blended) on 5,166; only 282 have a bit ≥ 4 (59 saturated, 20
  truncated, 219 incomplete aperture). `mask_flag_rSDSS` (MASK_FLAGS, Bonoli
  et al. 2021, Table 6): 1 = outside the window frame (normalised weight map
  < 85%, i.e. less exposure), 2 = bright star or near one, 4 = artefact (absent
  here). Bit 1 is set on 6,034 objects, bit 2 on 3,445. The severe cut
  (FLAGS bit ≥ 4, or MASK_FLAGS bit 2) removes 3,672 objects.
- Cross-tile duplicates within 1″: 12 pairs, so the tile split does not leak.
- Training pool after holdout and severe cut: 59,486 objects (3,873 of them
  only outside the window frame).

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
- Add a `quality` column in `dataset.py`, first match wins:
  `severe` (FLAGS bit ≥ 4, or MASK_FLAGS bit 2 = bright star) /
  `window` (MASK_FLAGS bit 1 only) / `blend` (FLAGS bits 1, 2 only) / `clean`.
- Resolved 2026-11-06 from Bonoli et al. (2021), Table 6: the bright-star bit
  means the photometry is contaminated, so it is severe. The window bit only
  means less exposure: shallower, but the quoted errors already carry that, so
  window objects stay in training as their own class. Evaluation reports them
  separately; if they reconstruct clearly worse, move them to `severe`.
- Training uses `clean` + `blend` + `window`. Predictions are still written for every
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
- `r_bb` is still noisy for the faintest objects (5% have S/N < 2.5). No
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
- `ε` is fixed in the config, per-band ε is an option. **Default 0.1**,
  chosen from the Fisher-spread check on the training tiles. The starting value
  of 0.01 was set for the unit-sphere normaliser, where |g| ≈ 1/√60; under
  `r_bb` the median |g| is ≈ 1.1, so the same ε is ~8× weaker. Rotation 0,
  training tiles, common-mode term on:

  | ε | objects holding 50% / 90% of the weight | stars' share (9% of objects) | cells where ε > v |
  |---|---|---|---|
  | 0.0001 | 0.7% / 5.5% | 0.80 | – |
  | 0.01 | 1.4% / 8.5% | 0.72 | – |
  | 0.03 | 2.2% / 14.7% | 0.61 | 1.3% |
  | 0.05 | 3.0% / 22.2% | 0.53 | 2.8% |
  | **0.1** | **5.5% / 38.3%** | **0.39** | **5.4%** |
  | 0.15 | 8.2% / 48.6% | 0.31 | 7.8% |
  | 0.2 | 10.9% / 55.2% | 0.26 | 10.4% |

  0.1 binds on about 5% of cells, like the old percentile rule, and the other
  rotations agree to within a percent. Stars still hold more weight than their
  number; ε is a natural knob for a later sweep.
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

## Results of the first sweep (2026-11-06)

20 VAE runs (latent 4/8/16 × 4 rotations; MLP encoder and `[g, mask]` at
d = 8 × 4 rotations), 50 epochs each, on CPU; baselines at the same sizes. The
full table is `reports/sweep_v2.md` (`make report`). Pooled test chi2/dof, mean
over rotations, ± half the range:

| model | d = 4 | d = 8 | d = 16 | active dims (d = 4/8/16) |
|---|---|---|---|---|
| VAE, conv encoder | 1.390 ±0.03 | 1.400 ±0.05 | 1.434 ±0.10 | 4.0 / 3.8 / 4.2 |
| VAE, `[g, mask]` | – | 1.385 ±0.04 | – | – / 4.5 / – |
| VAE, MLP encoder | – | 1.454 ±0.07 | – | – / 5.2 / – |
| weighted PCA | 1.363 ±0.02 | 1.219 ±0.02 | 0.972 ±0.01 | – |
| asinh PCA | 1.713 ±0.07 | 1.543 ±0.03 | 1.290 ±0.03 | – |

What it says:

1. **At β = 1 the VAE uses about 4 latent dimensions, whatever size it is
   given** (total KL ≈ 3 nats). It matches weighted PCA at d = 4 and loses to it
   at d = 8 and 16, where the extra dimensions stay at KL ≈ 0. This is the same
   partial posterior collapse the old project hit at β = 1. With per-band S/N
   ~1, the ELBO judges more latent information not worth its KL cost.
   Weighted PCA pays no such cost: it fits free coefficients to every test
   object, so it is an upper bound for an encoder, not an equal competitor.
2. **The convolutions help**: the MLP encoder is worse overall (1.454) and
   much worse on bright objects (S/N > 30: 1.30 against 0.95) and window-frame
   objects.
3. **Dropping `v` from the encoder costs nothing** (1.385 against 1.400, within
   the rotation spread) and removes most of the depth information from the
   encoder input.
4. **Window-frame objects reconstruct ~35% worse than clean ones** (chi2/dof
   ratio 1.34 for the VAE, 1.28 for weighted PCA), although their errors already
   carry the lower exposure. By the rule in A2 that argues for moving them to
   `severe`.
5. Asinh PCA is the weakest baseline at every size, as expected for a model
   without a noise model. Several runs pick their best epoch near epoch 50, and
   validation losses still move by a few units late in training (constant
   learning rate), so longer runs or a learning-rate schedule may help a little.

Choice for now: conv encoder, d = 8 (the default), since every size ends up
with ~4 active dimensions and d = 8 leaves room. The decisions that follow from
1, 3 and 4 are open items below.

## Open items

1. ~~Meaning of `mask_flag_rSDSS`~~ resolved (A2). Remaining: confirm that
   `window` objects reconstruct no worse than `clean` ones (C).
2. ~~ε default~~ resolved: 0.1 (B3). A later ε sweep is optional.
3. Whether to debias `r_bb` for faint objects (A4).
4. Conv widths and depth; whether `sig_logr` enters the encoder as a condition.
5. **Posterior collapse at β = 1** (result 1): decide between β < 1, a
   free-bits / KL floor per dimension, or accepting ~4 dimensions as what the
   data supports at this noise level.
6. **Default channels** (result 3): switch the default to `[g, mask]`?
7. **Window-frame objects** (result 4): move them to `severe` and rerun?
8. Learning-rate schedule / longer training (result 5).
9. Later goal, out of scope here: label-based inference on the holdout
   (redshift, class, anomalies) and morphology as an extra feature.
