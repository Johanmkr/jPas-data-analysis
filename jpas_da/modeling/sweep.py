"""Run the v2 sweeps locally: latent size, then ablations at the chosen size.

Each run is one ``python -m jpas_da.modeling.train`` process with ``--set``
overrides, writing ``models/sweep/<run>/vae.pt`` and
``reports/runs/<run>/metrics.json``. Runs execute ``--jobs`` at a time, each
limited to ``--threads`` CPU threads; a run whose metrics file already exists
is skipped, so an interrupted sweep resumes where it stopped.

Stages:

* ``size``: latent size 4 / 8 / 16 x every rotation (12 runs);
* ``ablation``: at ``--dim``, the flattened MLP encoder and the ``[g, mask]``
  channel set (which drops ``v`` and with it most of the depth information),
  x every rotation (8 runs).

The same run specs can go to the cluster through ``make job`` with
``TARGET=train`` and the overrides in ``SET``.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from typing import Annotated

from loguru import logger
import typer

from jpas_da.config import MODELS_DIR, PROJ_ROOT, REPORTS_DIR

app = typer.Typer()

ROTATIONS = (0, 1, 2, 3)
SIZES = (4, 8, 16)
RUNS_DIR = REPORTS_DIR / "runs"
SWEEP_MODELS_DIR = MODELS_DIR / "sweep"


def run_specs(stage: str, dim: int = 8) -> list[tuple[str, list[str]]]:
    """``(run name, overrides)`` for every run of a stage."""
    specs: list[tuple[str, list[str]]] = []
    if stage == "size":
        for d in SIZES:
            for k in ROTATIONS:
                specs.append((f"conv-d{d}-r{k}", [f"model.latent_dim={d}", f"data.rotation={k}"]))
    elif stage == "ablation":
        variants = {
            "mlp": ["model.encoder=mlp"],
            "gmask": ["model.channels=[g,mask]"],
        }
        for name, extra in variants.items():
            for k in ROTATIONS:
                overrides = [f"model.latent_dim={dim}", f"data.rotation={k}", *extra]
                specs.append((f"{name}-d{dim}-r{k}", overrides))
    else:
        raise ValueError(f"unknown stage {stage!r}; expected 'size' or 'ablation'")
    return specs


def metrics_path(run: str) -> Path:
    return RUNS_DIR / run / "metrics.json"


def _execute(run: str, overrides: list[str], threads: int) -> tuple[str, int]:
    out = metrics_path(run)
    if out.exists():
        logger.info(f"{run}: already done, skipping")
        return run, 0
    out.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "jpas_da.modeling.train",
        *[arg for o in overrides for arg in ("--set", o)],
        "--set",
        f"output.model_path={SWEEP_MODELS_DIR / run / 'vae.pt'}",
        "--set",
        f"output.metrics_path={out}",
    ]
    env = {
        **os.environ,
        "OMP_NUM_THREADS": str(threads),
        "MKL_NUM_THREADS": str(threads),
        # W&B (only active with WANDB_API_KEY set): one group per variant, run = directory.
        "WANDB_GROUP": os.environ.get("WANDB_GROUP", f"sweep-{run.split('-')[0]}"),
    }
    log = out.with_name("train.log")
    with log.open("w") as handle:
        code = subprocess.call(command, cwd=PROJ_ROOT, env=env, stdout=handle, stderr=handle)
    logger.info(f"{run}: {'done' if code == 0 else f'FAILED ({code}), see {log}'}")
    return run, code


@app.command()
def main(
    stage: Annotated[str, typer.Option("--stage")] = "size",
    dim: Annotated[int, typer.Option("--dim", help="Latent size for the ablations")] = 8,
    jobs: Annotated[int, typer.Option("--jobs")] = 3,
    threads: Annotated[int, typer.Option("--threads")] = 5,
) -> None:
    specs = run_specs(stage, dim)
    logger.info(f"stage {stage}: {len(specs)} runs, {jobs} at a time x {threads} threads")
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        results = list(pool.map(lambda spec: _execute(*spec, threads), specs))
    failed = [run for run, code in results if code != 0]
    if failed:
        raise SystemExit(f"{len(failed)} runs failed: {failed}")


if __name__ == "__main__":
    app()
