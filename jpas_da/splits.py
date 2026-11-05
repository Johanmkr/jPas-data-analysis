"""Grouped train/validation splits.

The unit of independence here is the pointing, not the object. miniJPAS is four
overlapping 0.27 deg^2 pointings; objects are never shared between them, but
neighbouring sources inside a pointing share sky background, crowding and the
survey's per-pointing photometric calibration. A random row split therefore
lets a model see, in training, sources whose noise it will be scored on.

``split_by_tile`` assigns each object a fold by holding out one pointing, which
is also the failure mode that matters in production: applying a model to a new
field. It returns plain integer arrays so :mod:`jpas_da.modeling.train` and
:mod:`jpas_da.modeling.predict` cannot disagree about which rows were held out.

The tile depths differ (median log-sigma 0.89 to 1.03 dex across the four
pointings), so a fold score is partly a measure of depth mismatch. That is the
intended reading, not an artefact to be normalised away.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from jpas_da.config import INTERIM_DATA_DIR

app = typer.Typer()

FOLDS_FILE = INTERIM_DATA_DIR / "folds.parquet"


def split_by_tile(
    tiles: np.ndarray, n_splits: int | None = None
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """Leave-one-tile-out fold assignment.

    Parameters
    ----------
    tiles
        Tile identifier per object, in object order.
    n_splits
        Number of folds. Defaults to one per distinct tile.

    Returns
    -------
    folds
        Fold index per object; ``-1`` where the tile was never held out.
    splits
        ``(train_index, test_index)`` per fold, as integer positions.
    """
    tiles = np.asarray(tiles)
    unique_tiles = np.unique(tiles)
    if len(unique_tiles) < 2:
        raise ValueError(f"need at least two tiles for a grouped split, got {len(unique_tiles)}")
    if n_splits is None:
        n_splits = len(unique_tiles)
    if not 2 <= n_splits <= len(unique_tiles):
        raise ValueError(f"n_splits must be in [2, {len(unique_tiles)}], got {n_splits}")

    folds = np.full(len(tiles), -1, dtype=np.int64)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold, tile in enumerate(unique_tiles[:n_splits]):
        held_out = tiles == tile
        folds[held_out] = fold
        splits.append((np.flatnonzero(~held_out), np.flatnonzero(held_out)))
        logger.info(
            f"fold {fold}: tile {tile} held out, {held_out.sum():,} test / "
            f"{(~held_out).sum():,} train objects"
        )
    return folds, splits


def build(objects_path: Path | None = None, output_path: Path = FOLDS_FILE) -> pd.DataFrame:
    """Write ``object_id`` -> fold, so every downstream step reads the same split."""
    objects_path = objects_path or INTERIM_DATA_DIR / "objects.parquet"
    objects = pd.read_parquet(objects_path, columns=["object_id", "tile_id"])
    folds, _ = split_by_tile(objects["tile_id"].to_numpy())
    assignment = pd.DataFrame(
        {"object_id": objects["object_id"], "tile_id": objects["tile_id"], "fold": folds}
    )
    if (assignment["fold"] < 0).any():
        raise ValueError("some objects were never assigned to a fold")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    assignment.to_parquet(output_path, index=False)
    logger.success(f"Wrote {output_path} ({assignment['fold'].nunique()} folds)")
    return assignment


@app.command()
def main(objects_path: Path | None = None, output_path: Path = FOLDS_FILE):
    build(objects_path, output_path)


if __name__ == "__main__":
    app()
