"""Encode every object with a trained VAE.

Writes one row per object -- training, validation, test, holdout and excluded
alike -- with its role in the checkpoint's rotation, so downstream analysis
(label-based inference on the holdout, anomaly searches) never has to re-run the
network:

``object_id``, ``tile_id``, ``role``, ``holdout``, ``quality``, ``logr``,
``sig_logr``, ``z_0..z_{d-1}`` (posterior means), ``logvar_0..`` and the
per-object ``chi2`` and ``dof`` of the deterministic reconstruction.

The checkpoint carries its own architecture, channels, conditions and noise
floor, so inference cannot silently disagree with training. The default output
sits next to the checkpoint: ``models/vae.pt -> models/vae_latents.parquet``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

from loguru import logger
import numpy as np
import pandas as pd
import torch
import typer

from jpas_da.config import MODELS_DIR
from jpas_da.features import MODEL_DATA_FILE
from jpas_da.modeling import data as model_data
from jpas_da.modeling import evaluate
from jpas_da.modeling.config import Config, _build
from jpas_da.modeling.losses import get_loss
from jpas_da.modeling.train import Tensors, resolve_device, run
from jpas_da.modeling.vae import build_model

app = typer.Typer()


def load_checkpoint(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model(checkpoint["model"]).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


def latents_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_latents.parquet")


@app.command()
def main(
    model_path: Annotated[Path, typer.Option("--model")] = MODELS_DIR / "vae.pt",
    data_path: Annotated[Path, typer.Option("--data")] = MODEL_DATA_FILE,
    output_path: Annotated[Path | None, typer.Option("--output")] = None,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4096,
    device_name: Annotated[str, typer.Option("--device")] = "auto",
) -> pd.DataFrame:
    device = resolve_device(device_name)
    model, ckpt = load_checkpoint(model_path, device)
    config = _build(Config, ckpt["config"], "checkpoint config")
    arrays = model_data.load(data_path)
    data = Tensors.build(arrays, tuple(ckpt["channels"]), tuple(ckpt["conditions"]), device)
    eps = torch.tensor(ckpt["eps"], dtype=torch.float32, device=device)

    index = np.arange(len(arrays["g"]))
    out = run(model, get_loss(config.loss.name), config, data, index, eps, batch_size)
    chi2, dof = evaluate.chi2_per_object(arrays["g"], out["g_hat"], out["w2"], arrays["mask"])

    frame = {
        "object_id": arrays["object_id"],
        "tile_id": arrays["tile_id"],
        "role": model_data.roles(arrays, int(ckpt["rotation"])).astype(str),
        "holdout": arrays["holdout"],
        "quality": arrays["quality"],
        "logr": arrays["logr"],
        "sig_logr": arrays["sig_logr"],
        "chi2": chi2,
        "dof": dof,
    }
    for i in range(out["mu_z"].shape[1]):
        frame[f"z_{i}"] = out["mu_z"][:, i]
    for i in range(out["logvar_z"].shape[1]):
        frame[f"logvar_{i}"] = out["logvar_z"][:, i]
    table = pd.DataFrame(frame)

    output_path = output_path or latents_path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(output_path, index=False)
    logger.success(f"Wrote {output_path} ({len(table):,} objects, {out['mu_z'].shape[1]} latents)")
    print(json.dumps({"output": str(output_path), "n_objects": len(table)}))
    return table


if __name__ == "__main__":
    app()
