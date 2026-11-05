"""Train the SED-shape VAE.

The loop is one epoch of minibatches over the training fold, validating on the
held-out tile each epoch. It writes two artefacts: a checkpoint (model weights
plus the arrays needed to rebuild it) and ``reports/metrics.json`` with the
per-epoch history, the fitted noise floor and the resolved configuration.

All modelling choices come from ``configs/vae.yaml``; ``--set key=value``
overrides any of them without editing the file, and ``--debug`` trains on a
small subset for a couple of epochs so the pipeline can be smoke-tested.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import random
import time
from typing import Annotated

from loguru import logger
import numpy as np
import torch
import typer

from jpas_da.config import PROJ_ROOT
from jpas_da.modeling import data as model_data
from jpas_da.modeling.config import Config, dump_config, load_config
from jpas_da.modeling.losses import get_loss
from jpas_da.modeling.vae import VAE, normalise_shape

app = typer.Typer()

DEFAULT_CONFIG = PROJ_ROOT / "configs" / "vae.yaml"


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("device 'cuda' requested but no CUDA device is visible")
    return device


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def noise_floor(v: torch.Tensor, mask: torch.Tensor, percentile: float) -> torch.Tensor:
    """Per-band additive error floor from the training fold.

    The archive's quoted error is purely relative, so ``v`` vanishes with the
    flux and a near-zero-flux band would claim infinite precision. The floor is
    a percentile of the *finite* weighted errors in each band, computed on the
    training rows only, and it is fixed for the run -- never learned.
    """
    bands = int(v.shape[1])
    floored = v.new_empty(bands)
    for band in range(bands):
        column = v[:, band][mask[:, band] > 0]
        column = column[torch.isfinite(column)]
        if column.numel() == 0:
            raise ValueError(f"no defined pixels in band {band}; cannot set a floor")
        floored[band] = torch.quantile(column, percentile / 100.0)
    if not torch.isfinite(floored).all() or (floored <= 0).any():
        raise ValueError("noise floor must be finite and positive in every band")
    return floored


def beta_at(step: int, total_steps: int, config: Config) -> float:
    """Linear warm-up of the KL weight over the first configured fraction."""
    warmup = max(1, round(total_steps * config.loss.beta_warmup_fraction))
    if step >= warmup:
        return config.loss.beta
    return config.loss.beta * step / warmup


def tensors_for(arrays: dict[str, np.ndarray], config: Config) -> tuple[np.ndarray, np.ndarray]:
    """Stack the configured channels, keeping ``mask`` as the trailing channel."""
    channels = [arrays[name].astype(np.float32) for name in config.model.channels]
    x = np.stack(channels, axis=1)
    mask = arrays["mask"].astype(np.float32)
    return x, mask


def batch_losses(
    model: VAE,
    loss_fn,
    config: Config,
    x: torch.Tensor,
    g: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    logr: torch.Tensor,
    sig_logr: torch.Tensor,
    s_floor: torch.Tensor,
    beta: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    out = model(x)
    mu_g = normalise_shape(out["mu_g"], mask)
    rec = loss_fn(g, v, mask, mu_g, s_floor, dof=config.loss.student_t_dof).mean()

    mu_z, logvar_z = out["mu_z"], out["logvar_z"]
    kl = 0.5 * (mu_z**2 + torch.exp(logvar_z) - 1.0 - logvar_z).sum(-1).mean()

    loss = rec + beta * kl
    parts = {"reconstruction": float(rec.detach()), "kl": float(kl.detach())}

    if out["l_mu"] is not None:
        scale = (
            0.5
            * (logr - out["l_mu"].squeeze(-1)) ** 2
            / (sig_logr**2 + config.loss.systematic_scale**2)
        )
        scale = scale.mean()
        loss = loss + scale
        parts["scale"] = float(scale.detach())

    parts["loss"] = float(loss.detach())
    return loss, parts


def evaluate(
    model: VAE,
    loss_fn,
    config: Config,
    data: dict[str, torch.Tensor],
    index: np.ndarray,
    s_floor: torch.Tensor,
    beta: float,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    seen = 0
    with torch.no_grad():
        for start in range(0, len(index), batch_size):
            rows = index[start : start + batch_size]
            _, parts = batch_losses(
                model,
                loss_fn,
                config,
                *(tensor[rows] for tensor in data["inputs"]),
                s_floor=s_floor,
                beta=beta,
            )
            for key, value in parts.items():
                totals[key] = totals.get(key, 0.0) + value * len(rows)
            seen += len(rows)
    return {key: value / seen for key, value in totals.items()}


@app.command()
def main(
    config_path: Annotated[
        Path, typer.Option("--config", help="YAML config file")
    ] = DEFAULT_CONFIG,
    overrides: Annotated[
        list[str] | None, typer.Option("--set", help="Config override as dotted.key=value")
    ] = None,
    debug: Annotated[bool, typer.Option("--debug", help="Small subset, few epochs")] = False,
    seed: Annotated[int | None, typer.Option("--seed", help="Override train.seed")] = None,
) -> dict[str, object]:
    """Train the VAE. ``--set dotted.key=value`` overrides the YAML."""
    config = load_config(config_path, list(overrides or []))
    if seed is not None:
        config.train.seed = seed
    if debug:
        config.train.epochs = config.train.debug_epochs
    set_seed(config.train.seed)

    device = resolve_device(config.train.device)
    arrays = model_data.load(config.data.path)
    x_np, mask_np = tensors_for(arrays, config)

    folds = arrays["fold"]
    fold_ids = np.unique(folds)
    if config.data.fold not in fold_ids:
        raise ValueError(f"fold {config.data.fold} not in {fold_ids.tolist()}")
    train_index = np.flatnonzero(folds != config.data.fold)
    val_index = np.flatnonzero(folds == config.data.fold)
    if debug:
        if config.train.debug_objects > len(train_index):
            raise ValueError(
                f"debug_objects={config.train.debug_objects} exceeds the "
                f"{len(train_index)} training objects"
            )
        rng = np.random.default_rng(config.train.seed)
        train_index = np.sort(rng.choice(train_index, config.train.debug_objects, replace=False))

    inputs = (
        torch.from_numpy(x_np).to(device),
        torch.from_numpy(arrays["g"]).to(device),
        torch.from_numpy(arrays["v"]).to(device),
        torch.from_numpy(mask_np).to(device),
        torch.from_numpy(arrays["logr"]).to(device),
        torch.from_numpy(arrays["sig_logr"]).to(device),
    )
    data = {"inputs": inputs}

    s_floor = noise_floor(
        inputs[2][train_index], inputs[3][train_index], config.loss.floor_percentile
    )
    logger.info(
        f"per-band noise floor (p{config.loss.floor_percentile:g} of v): "
        f"min {float(s_floor.min()):.4f}, median {float(s_floor.median()):.4f}, "
        f"max {float(s_floor.max()):.4f}"
    )

    n_bands = int(arrays["n_bands"])
    model = VAE(
        n_bands=n_bands,
        n_channels=len(config.model.channels),
        latent_dim=config.model.latent_dim,
        hidden=config.model.hidden,
        kernel_size=config.model.kernel_size,
        scale_head=config.model.scale_head,
        dropout=config.model.dropout,
    ).to(device)
    loss_fn = get_loss(config.loss.name)
    optimiser = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.learning_rate,
        weight_decay=config.train.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_index) / config.train.batch_size)
    total_steps = steps_per_epoch * config.train.epochs
    logger.info(
        f"{config.train.epochs} epochs, {len(train_index):,} train / "
        f"{len(val_index):,} val objects, {steps_per_epoch} steps/epoch on {device}"
    )

    history: list[dict[str, float]] = []
    step = 0
    started = time.time()
    rng = np.random.default_rng(config.train.seed + 1)
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        order = rng.permutation(train_index)
        running: dict[str, float] = {}
        for batch_start in range(0, len(order), config.train.batch_size):
            rows = order[batch_start : batch_start + config.train.batch_size]
            beta = beta_at(step, total_steps, config)
            loss, parts = batch_losses(
                model,
                loss_fn,
                config,
                *(tensor[rows] for tensor in inputs),
                s_floor=s_floor,
                beta=beta,
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimiser.step()
            step += 1
            for key, value in parts.items():
                running[key] = running.get(key, 0.0) + value
            if config.train.log_every and step % config.train.log_every == 0:
                logger.info(
                    f"epoch {epoch} step {step}/{total_steps} "
                    f"loss {parts['loss']:.3f} rec {parts['reconstruction']:.3f} "
                    f"kl {parts['kl']:.3f} beta {beta:.2f}"
                )

        train_metrics = {key: value / max(1, steps_per_epoch) for key, value in running.items()}
        val_metrics = evaluate(
            model,
            loss_fn,
            config,
            data,
            val_index,
            s_floor,
            config.loss.beta,
            config.train.batch_size,
        )
        record = {
            "epoch": epoch,
            "beta": beta_at(step, total_steps, config),
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(record)
        logger.info(
            f"epoch {epoch}: train {record['train_loss']:.3f} "
            f"(rec {record['train_reconstruction']:.3f}, kl {record['train_kl']:.3f}) | "
            f"val {record['val_loss']:.3f}"
        )

    config.output.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model": {
                "n_bands": n_bands,
                "n_channels": len(config.model.channels),
                "latent_dim": config.model.latent_dim,
                "hidden": list(config.model.hidden),
                "scale_head": config.model.scale_head,
            },
            "channels": list(config.model.channels),
            "s_floor": s_floor.cpu(),
            "config": config.to_dict(),
        },
        config.output.model_path,
    )

    final = history[-1]
    metrics = {
        "config": config.to_dict(),
        "config_source": str(config_path) if config_path else None,
        "overrides": list(overrides or []),
        "debug": debug,
        "device": str(device),
        "n_objects": {"train": len(train_index), "val": len(val_index)},
        "n_bands": n_bands,
        "channels": list(config.model.channels),
        "noise_floor": [float(value) for value in s_floor.cpu()],
        "history": history,
        "final": final,
        "elapsed_seconds": round(time.time() - started, 2),
        "model_path": str(config.output.model_path),
    }
    if not all(math.isfinite(final[f"val_{key}"]) for key in ("loss", "reconstruction")):
        raise RuntimeError(f"non-finite validation loss after training: {final}")

    config.output.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.output.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    logger.success(
        f"Wrote {config.output.model_path} and {config.output.metrics_path} "
        f"(val loss {final['val_loss']:.3f})"
    )
    if debug:
        logger.info("config used:\n" + dump_config(config))
    return metrics


if __name__ == "__main__":
    app()
