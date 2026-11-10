"""Train / validation / test roles by tile rotation, plus the inference holdout.

The unit of independence is the pointing, not the object. miniJPAS has four
pointings; neighbouring sources inside one share sky background, crowding and
the per-pointing calibration, so a random row split lets a model see, in
training, sources whose noise it is scored on. Holding out whole tiles is also
the case that matters in practice: applying the model to a new field.

With tiles in sorted order ``T`` and ``n = len(T)``, rotation ``k`` uses::

    test  = T[k]
    val   = T[(k + 1) % n]
    train = every other tile

so each tile is the test tile exactly once over ``n`` rotations, and the test
tile is never used for model selection. Roles are derived from ``k`` in code;
nothing about a rotation is stored that could drift.

Two kinds of object get no role:

* ``holdout`` -- the SDSS-spectroscopic objects, kept for label-based
  inference (see :func:`jpas_da.dataset.label_flags`);
* ``excluded`` -- ``quality == "severe"`` (compromised photometry; see
  :func:`jpas_da.dataset.quality_class`).

The tile depths differ (median log-sigma 0.89 to 1.03 dex), so a test score is
partly a measure of depth mismatch. That is the intended reading.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from jpas_da.config import INTERIM_DATA_DIR
from jpas_da.dataset import TRAINING_QUALITY

app = typer.Typer()

SPLITS_FILE = INTERIM_DATA_DIR / "splits.parquet"
ROLES = ("train", "val", "test")


def rotation_tiles(tiles: np.ndarray, rotation: int) -> dict[str, list[int]]:
    """Which tiles play which role in ``rotation``."""
    unique = sorted(int(t) for t in np.unique(tiles))
    n = len(unique)
    if n < 3:
        raise ValueError(f"need at least three tiles for train/val/test, got {n}")
    if not 0 <= rotation < n:
        raise ValueError(f"rotation must be in [0, {n - 1}], got {rotation}")
    test, val = unique[rotation], unique[(rotation + 1) % n]
    return {"train": [t for t in unique if t not in (test, val)], "val": [val], "test": [test]}


def assign_roles(
    tile_id: np.ndarray, holdout: np.ndarray, quality: np.ndarray, rotation: int
) -> np.ndarray:
    """One role per object: ``train``/``val``/``test``/``holdout``/``excluded``."""
    tile_id = np.asarray(tile_id)
    roles = np.empty(len(tile_id), dtype=object)
    for role, tiles in rotation_tiles(tile_id, rotation).items():
        roles[np.isin(tile_id, tiles)] = role
    roles[~np.isin(np.asarray(quality), TRAINING_QUALITY)] = "excluded"
    roles[np.asarray(holdout, dtype=bool)] = "holdout"
    return roles


def n_rotations(tile_id: np.ndarray) -> int:
    return len(np.unique(tile_id))


def build(objects_path: Path | None = None, output_path: Path = SPLITS_FILE) -> pd.DataFrame:
    """Write ``object_id``, ``tile_id``, ``holdout``, ``quality`` and ``role_<k>``.

    The ``role_<k>`` columns are a convenience for analysis joins; training
    recomputes them with :func:`assign_roles`, so they cannot disagree.
    """
    objects_path = objects_path or INTERIM_DATA_DIR / "objects.parquet"
    if not Path(objects_path).exists():
        raise FileNotFoundError(
            f"interim table not found: {objects_path}\n"
            "It is built from the raw export by `make interim` (or `make data`), which\n"
            "needs data/raw/miniJpas_fluxes_errors_redshifts_full.csv; see\n"
            "references/getting-started.md, section 2."
        )
    objects = pd.read_parquet(objects_path, columns=["object_id", "tile_id", "holdout", "quality"])
    table = objects.copy()
    for k in range(n_rotations(objects["tile_id"])):
        roles = assign_roles(objects["tile_id"], objects["holdout"], objects["quality"], k)
        table[f"role_{k}"] = roles
        counts = pd.Series(roles).value_counts()
        tiles = rotation_tiles(objects["tile_id"].to_numpy(), k)
        logger.info(
            f"rotation {k}: test {tiles['test']}, val {tiles['val']}, train {tiles['train']} | "
            + ", ".join(f"{r} {int(counts.get(r, 0)):,}" for r in (*ROLES, "holdout", "excluded"))
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    logger.success(f"Wrote {output_path}")
    return table


@app.command()
def main(objects_path: Path | None = None, output_path: Path = SPLITS_FILE):
    build(objects_path, output_path)


if __name__ == "__main__":
    app()
