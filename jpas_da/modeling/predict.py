"""Encode objects with a trained VAE: latent means, brightness and reconstruction.

Writes one row per object (``object_id``, ``tile_id``, ``fold``, ``z_*``,
``logr``, ``l_mu``, per-object losses) so downstream analysis never has to
re-run the network. The checkpoint carries its own architecture and channel
list, so inference cannot silently disagree with training.
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

from jpas_da.config import MODELS_DIR, PROCESSED_DATA_DIR
from jpas_da.modeling import data as model_data
from jpas_da.modeling.losses import get_loss
from jpas_da.modeling.train import resolve_device
from jpas_da.modeling.vae import VAE, normalise_shape

app = typer.Typer()

PREDICTIONS_FILE = PROCESSED_DATA_DIR / "vae_latents.parquet"


def load_checkpoint(path: Path, device: torch.device) -> tuple[VAE, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    spec = checkpoint["model"]
    model = VAE(
        n_bands=spec["n_bands"],
        n_channels=spec["n_channels"],
        latent_dim=spec["latent_dim"],
        hidden=tuple(spec["hidden"]),
        scale_head=spec["scale_head"],
    ).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint


@app.command()
def main(
    model_path: Annotated[Path, typer.Option("--model")] = MODELS_DIR / "vae.pt",
    data_path: Annotated[Path, typer.Option("--data")] = PROCESSED_DATA_DIR / "vae_inputs.npz",
    output_path: Annotated[Path, typer.Option("--output")] = PREDICTIONS_FILE,
    batch_size: Annotated[int, typer.Option("--batch-size")] = 4096,
    device_name: Annotated[str, typer.Option("--device")] = "auto",
) -> None:
    device = resolve_device(device_name)
    model, checkpoint = load_checkpoint(model_path, device)
    arrays = model_data.load(data_path)

    channels = checkpoint["channels"]
    x = np.stack([arrays[name].astype(np.float32) for name in channels], axis=1)
    mask = torch.from_numpy(arrays["mask"].astype(np.float32)).to(device)
    s_floor = checkpoint["s_floor"].to(device)
    config = checkpoint["config"]
    loss_fn = get_loss(config["loss"]["name"])

    z_chunks: list[np.ndarray] = []
    l_mu_chunks: list[np.ndarray] = []
    rec_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            rows = slice(start, start + batch_size)
            batch_x = torch.from_numpy(x[rows]).to(device)
            out = model(batch_x)
            z_chunks.append(out["mu_z"].cpu().numpy())
            g = torch.from_numpy(arrays["g"][rows]).to(device)
            v = torch.from_numpy(arrays["v"][rows]).to(device)
            batch_mask = mask[rows]
            mu = normalise_shape(out["mu_g"], batch_mask)
            rec = loss_fn(
                g,
                v,
                batch_mask,
                mu,
                s_floor,
                dof=config["loss"]["student_t_dof"],
            )
            rec_chunks.append(rec.cpu().numpy())
            if out["l_mu"] is not None:
                l_mu_chunks.append(out["l_mu"].squeeze(-1).cpu().numpy())

    z = np.concatenate(z_chunks)
    frame = {
        "object_id": arrays["object_id"],
        "tile_id": arrays["tile_id"],
        "logr": arrays["logr"],
        "reconstruction": np.concatenate(rec_chunks),
    }
    if "fold" in arrays:
        frame["fold"] = arrays["fold"]
    for index in range(z.shape[1]):
        frame[f"z_{index}"] = z[:, index]
    if l_mu_chunks:
        frame["l_mu"] = np.concatenate(l_mu_chunks)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame).to_parquet(output_path, index=False)
    logger.success(f"Wrote {output_path} ({len(z):,} objects, {z.shape[1]} latents)")
    print(json.dumps({"output": str(output_path), "n_objects": len(z), "latent_dim": z.shape[1]}))


if __name__ == "__main__":
    app()
