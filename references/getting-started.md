# Getting started

From a fresh clone to a trained checkpoint. Every command below is a `make`
target; the Python entry point behind each one is listed at the end.

## Requirements

| | Version | Notes |
|---|---|---|
| Python | 3.10 (`~=3.10.0`) | pinned in `pyproject.toml` |
| [uv](https://docs.astral.sh/uv/) | 0.12 or newer | resolves `uv.lock`; the Makefile calls `uv run` |
| GNU make | 4.x | the default goal is `make help` |
| Docker | optional | only for `make image` / `make run RUNNER=docker` |

No GPU is needed. `torch` resolves from PyPI to the CUDA build on linux-x86_64,
which imports and runs on a CPU-only host. Timings measured on this machine
(22 CPU cores): `make data` 4 s, `make train-debug` 6 s, one full epoch on a
rotation's 28k-31k objects ~2 s, so `make train` (50 epochs) is roughly two
minutes on CPU. A GPU makes each epoch faster but is never required.

## 1. Environment

```bash
git clone git@github.com:Johanmkr/jPas-data-analysis.git
cd jPas-data-analysis
uv sync                       # or: make requirements
uv run pytest -q              # 113 tests; 1 skips without the raw CSV
```

`uv sync` installs the locked environment into `.venv/`. `make` uses
`uv run python`, so no activation step is needed.

## 2. Getting the data

**`data/` is git-ignored: the repository ships code, configs and docs, not the
catalogue.** The pipeline starts from

```
data/raw/miniJpas_fluxes_errors_redshifts_full.csv   64,327 rows x 137 cols, 78 MB
```

which is produced by `references/queries/minijpas_flux_with_errors_full.txt`
against the miniJPAS archive — `references/queries/README.md` (page *Data
sources* in the docs site) lists the tables it touches and the sha256 to verify a
copy. Put the file at exactly that path and check it:

```bash
sha256sum data/raw/miniJpas_fluxes_errors_redshifts_full.csv
# e747b331be4a27657bb2afd98ebdb71dc5fd52d0ead76b0431ac68d082deace5
```

Without the file, `make data` and everything downstream stops with a
`FileNotFoundError` that names this path and points at the query; the other
`make` targets do the same. Nothing else in the repository needs it: the tests
skip the one integration case, and the docs still build.

## 3. Build the model inputs

```bash
make data
```

Two seconds for the interim table and two for the arrays. It writes:

| File | Contents |
|---|---|
| `data/interim/objects.parquet` | one row per object: flux, absolute sigma, mask, metadata, `holdout`, `quality` (201 columns) |
| `data/interim/splits.parquet` | role of every object in each of the 4 tile rotations |
| `data/processed/vae_inputs.npz` | `g`, `v`, `mask`, `logr`, `sig_logr` + identifiers |

Each stage also runs alone: `make interim`, `make splits`, `make features`.
Every stage is idempotent, and rebuilding produces an identical
`data/interim/objects.parquet` (checked with `DataFrame.equals`).

## 4. Train and inspect

```bash
make train-debug      # 1,024 objects, 2 epochs: proves the pipeline runs (~12 s incl. make data)
make train            # rotation 0, configs/vae.yaml, 50 epochs
make predict          # every object (incl. holdout) -> models/vae_latents.parquet
make report           # reports/runs/ + baselines -> reports/sweep_v2.md
```

| Output | What it is |
|---|---|
| `models/vae.pt` | best-by-validation checkpoint: weights, architecture, channels, `eps`, rotation, resolved config |
| `models/vae_last.pt` | the last epoch |
| `reports/metrics.json` | per-epoch history, KL per latent dimension, χ²/dof on the validation and test tiles |
| `models/vae_latents.parquet` | `z_0…z_{d-1}`, `logvar_*`, `chi2`, `dof` per object (from `make predict`) |

## 5. Explore

```bash
make notebook NOTEBOOK=notebooks/explore_data.py
```

Runs marimo inside the project environment. The notebook loads the real files and
walks from the SQL export to the arrays the VAE optimises; it executes in a few
seconds and needs `make data` first.

## What to read, in order

1. `README.md` — the pipeline end to end, one screen per topic.
2. `references/queries/README.md` — where the CSV comes from, and the query.
3. `references/data_dictionary.md` — every column and every count.
4. `references/pipeline-v2-plan.md` — why each design choice was made, with the
   measurements behind it and the alternatives that lost.
5. The module docstrings: `jpas_da/dataset.py`, `splits.py`, `features.py`,
   then `modeling/data.py`, `vae.py`, `losses.py`, `train.py`.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `FileNotFoundError: .../miniJpas_fluxes_errors_redshifts_full.csv` | the raw export is missing; see step 2 |
| `raw export is missing N expected columns` | a different export, or the CSV header line was stripped when saving |
| `uv: command not found` | install uv, or run `PYTHON_INTERPRETER=python make …` inside an activated venv |
| `ValueError: device 'cuda' requested but no CUDA device is visible` | `train.device` is `cuda` in a config or `SET`; use `auto` |
| `make data` re-runs every time | it is `.PHONY` on purpose: a few seconds, and no stale derived table survives a code change |
| `interim table not found: ...` | `data/interim/objects.parquet` is missing; run `make interim` (which needs the raw export) |

## Entry points

| Make target | Module |
|---|---|
| `make interim` | `python -m jpas_da.dataset` |
| `make splits` | `python -m jpas_da.splits` |
| `make features` | `python -m jpas_da.features` |
| `make train`, `train-debug` | `python -m jpas_da.modeling.train` |
| `make predict` | `python -m jpas_da.modeling.predict` |
| `make baselines` | `python -m jpas_da.modeling.baselines` |
| `make sweep` | `python -m jpas_da.modeling.sweep` |
| `make report` | `python -m jpas_da.modeling.report` |
| `make job`, `make submit` | `python -m jpas_da.cluster` |

Each module takes `--help`; `train` additionally takes `--set dotted.key=value`
and `--debug`.
