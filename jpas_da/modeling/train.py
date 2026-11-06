"""Train the conditional SED VAE on one tile rotation.

One run = one rotation (``data.rotation``): it trains on two tiles, selects the
best epoch on the validation tile, and reports on the test tile, which is never
used for any choice. The holdout (SDSS-spectroscopic objects) and the
``severe`` quality class never enter.

Artefacts:

* ``output.model_path`` -- the best-by-validation checkpoint (weights,
  architecture spec, channels, conditions, eps, rotation, resolved config);
  ``<stem>_last.pt`` next to it holds the last epoch.
* ``output.metrics_path`` -- per-epoch history (training loss with sampling;
  validation loss, reconstruction and KL per dimension without sampling), the
  best epoch, and the full :mod:`jpas_da.modeling.evaluate` reports on the
  validation and test tiles.

All modelling choices come from ``configs/vae.yaml``; ``--set key=value``
overrides any of them, and ``--debug`` trains on a small subset for a couple of
epochs so the pipeline can be smoke-tested.

Weights & Biases logging is optional and follows the old repo's pattern: it is
on only when ``wandb`` is importable, ``WANDB_API_KEY`` is set and
``WANDB_MODE`` is not ``disabled``/``offline``/``dryrun``, so local runs and
tests without a key never log. ``WANDB_PROJECT``, ``WANDB_ENTITY``,
``WANDB_RUN_NAME`` and ``WANDB_GROUP`` choose where a run goes (``--wandb-project``
and ``--wandb-entity`` override the first two). Each epoch logs the training and
validation losses, beta, the learning rate and the validation KL per latent
dimension; the end of the run writes the best epoch and the test/validation
reports to the run summary. The JSON metrics and checkpoints are written either
way: W&B is for monitoring, not the record.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Annotated, Any

from loguru import logger
import numpy as np
import torch
import typer

from jpas_da.bands import N_BANDS
from jpas_da.config import PROJ_ROOT
from jpas_da.modeling import data as model_data
from jpas_da.modeling import evaluate
from jpas_da.modeling.config import Config, dump_config, load_config
from jpas_da.modeling.losses import get_loss, noise_variance
from jpas_da.modeling.vae import VAE, build_model, kl_per_dim

app = typer.Typer()

DEFAULT_CONFIG = PROJ_ROOT / "configs" / "vae.yaml"
DEBUG_EVAL_OBJECTS = 2048
"""Validation/test objects scored per epoch in ``--debug`` runs."""

WANDB_PROJECT = "jpas-da"
WANDB_ENTITY = "johan-m-kroken-uit-norges-arktiske-universitet"


# --------------------------------------------------------------------------- #
# Weights & Biases
# --------------------------------------------------------------------------- #
def wandb_enabled() -> bool:
    """Log only with ``wandb`` installed, an API key, and an online mode."""
    try:
        import wandb  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("WANDB_MODE", "").lower() in ("disabled", "dryrun", "offline"):
        return False
    return bool(os.environ.get("WANDB_API_KEY"))


def run_name(config: Config, debug: bool) -> str:
    """``WANDB_RUN_NAME``, else the run directory a sweep gives, else a description."""
    if os.environ.get("WANDB_RUN_NAME"):
        return os.environ["WANDB_RUN_NAME"]
    parent = config.output.metrics_path.parent
    if parent.name not in ("reports", ""):
        return parent.name
    name = f"{config.model.encoder}-d{config.model.latent_dim}-r{config.data.rotation}"
    return f"{name}-debug" if debug else name


class Tracker:
    """Thin wrapper so the training loop never checks whether W&B is on."""

    def __init__(self, run: Any = None) -> None:
        self.run = run

    def log(self, values: dict[str, float], step: int) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def summary(self, values: dict[str, object]) -> None:
        if self.run is not None:
            self.run.summary.update(values)


def start_wandb(config: Config, debug: bool, overrides: list[str]) -> tuple[Tracker, bool]:
    """``(tracker, started_here)``; a no-op tracker when logging is off.

    If a run is already active (e.g. a ``wandb agent`` started it), log into it
    and leave finishing it to its owner.
    """
    if not wandb_enabled():
        return Tracker(), False
    import wandb

    resolved = {**config.to_dict(), "overrides": overrides, "debug": debug}
    if wandb.run is not None:
        wandb.config.update(resolved, allow_val_change=True)
        return Tracker(wandb.run), False
    run = wandb.init(
        project=os.environ.get("WANDB_PROJECT", WANDB_PROJECT),
        entity=os.environ.get("WANDB_ENTITY", WANDB_ENTITY),
        name=run_name(config, debug),
        group=os.environ.get("WANDB_GROUP"),
        job_type="debug" if debug else "train",
        tags=[
            config.model.encoder,
            f"d{config.model.latent_dim}",
            f"rotation-{config.data.rotation}",
        ],
        config=resolved,
    )
    logger.info(f"W&B run {run.name}: {run.url}")
    return Tracker(run), True


@contextmanager
def tracking(config: Config, debug: bool, overrides: list[str]) -> Iterator[Tracker]:
    """Start W&B if enabled; finish the run, marked failed if training raised."""
    tracker, started = start_wandb(config, debug, overrides)
    try:
        yield tracker
    except BaseException:
        if started:
            tracker.run.finish(exit_code=1)
        raise
    if started:
        tracker.run.finish()


def epoch_log(record: dict[str, object], lr: float) -> dict[str, float]:
    """Flatten one history record for W&B: scalars plus ``val_kl_dim/<i>``."""
    values = {k: v for k, v in record.items() if isinstance(v, (int, float))}
    values["lr"] = lr
    kl_dims = record.get("val_kl_per_dim", [])
    for i, kl in enumerate(kl_dims):
        values[f"val_kl_dim/{i}"] = kl
    values["val_active_dims"] = sum(kl > evaluate.ACTIVE_KL for kl in kl_dims)
    return values


def summary_values(metrics: dict[str, Any]) -> dict[str, object]:
    """The numbers to compare runs by, flattened for the W&B run summary."""
    values: dict[str, object] = {"best_epoch": metrics["best_epoch"]}
    for role in ("val", "test"):
        rec = metrics[role]["reconstruction"]
        values[f"{role}/nll"] = metrics[role]["nll"]
        values[f"{role}/chi2_dof_pooled"] = rec["overall"]["chi2_dof_pooled"]
        values[f"{role}/chi2_dof_median"] = rec["overall"]["chi2_dof_median"]
        values[f"{role}/active_dims"] = metrics[role]["latent"]["active_dims"]
        values[f"{role}/kl_total"] = metrics[role]["latent"]["kl_total"]
        for group in ("by_snr", "by_quality"):
            for key, summary in rec[group].items():
                if "chi2_dof_pooled" in summary:
                    values[f"{role}/{group}/{key}"] = summary["chi2_dof_pooled"]
    return values


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


def last_path(model_path: Path) -> Path:
    return model_path.with_name(f"{model_path.stem}_last{model_path.suffix}")


@dataclass
class Tensors:
    """Every per-object tensor the model and the loss need, on one device."""

    narrow: torch.Tensor
    other: torch.Tensor
    cond: torch.Tensor
    g: torch.Tensor
    v: torch.Tensor
    mask: torch.Tensor
    sig_logr: torch.Tensor

    @classmethod
    def build(
        cls,
        arrays: dict[str, np.ndarray],
        channels: tuple[str, ...],
        conditions: tuple[str, ...],
        device: torch.device,
    ) -> Tensors:
        blocks = model_data.model_inputs(arrays, channels, conditions)
        values = {
            **blocks,
            "g": arrays["g"].astype(np.float32),
            "v": arrays["v"].astype(np.float32),
            "mask": arrays["mask"].astype(np.float32),
            "sig_logr": arrays["sig_logr"].astype(np.float32),
        }
        return cls(
            **{k: torch.from_numpy(np.ascontiguousarray(a)).to(device) for k, a in values.items()}
        )

    def rows(self, index) -> Tensors:
        return Tensors(**{k: getattr(self, k)[index] for k in self.__dataclass_fields__})


def floor_tensor(config: Config, device: torch.device) -> torch.Tensor:
    eps = config.loss.eps_per_band or (config.loss.eps,)
    return torch.tensor(eps, dtype=torch.float32, device=device)


def batch_terms(
    model: VAE, loss_fn, config: Config, batch: Tensors, eps: torch.Tensor, sample: bool
) -> dict[str, torch.Tensor]:
    """Per-object reconstruction term, KL per dimension and model output."""
    out = model(batch.narrow, batch.other, batch.cond, sample=sample)
    w2 = noise_variance(batch.g, batch.v, batch.mask, batch.sig_logr, eps, config.loss.common_mode)
    rec = loss_fn(batch.g, out["g_hat"], w2, batch.mask, dof=config.loss.student_t_dof)
    return {"rec": rec, "kl": kl_per_dim(out["mu_z"], out["logvar_z"]), "w2": w2, **out}


def beta_at(step: int, total_steps: int, config: Config) -> float:
    """Linear warm-up of the KL weight over the first configured fraction."""
    warmup = max(1, round(total_steps * config.loss.beta_warmup_fraction))
    if step >= warmup:
        return config.loss.beta
    return config.loss.beta * step / warmup


@torch.no_grad()
def run(
    model: VAE,
    loss_fn,
    config: Config,
    data: Tensors,
    index: np.ndarray,
    eps: torch.Tensor,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Deterministic pass (``z = mu_z``) over ``index``; numpy outputs per object."""
    model.eval()
    keys = ("rec", "kl", "w2", "mu_z", "logvar_z", "g_hat")
    chunks: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    for start in range(0, len(index), batch_size):
        terms = batch_terms(
            model, loss_fn, config, data.rows(index[start : start + batch_size]), eps, False
        )
        for key in keys:
            chunks[key].append(terms[key].cpu().numpy())
    if not len(index):
        return {}
    return {key: np.concatenate(parts) for key, parts in chunks.items()}


def epoch_summary(result: dict[str, np.ndarray], beta: float) -> dict[str, object]:
    rec = float(result["rec"].mean())
    kl_dims = result["kl"].mean(axis=0)
    return {
        "loss": rec + beta * float(kl_dims.sum()),
        "reconstruction": rec,
        "kl": float(kl_dims.sum()),
        "kl_per_dim": [float(x) for x in kl_dims],
    }


def report(
    result: dict[str, np.ndarray], data: Tensors, arrays: dict, index: np.ndarray
) -> dict[str, object]:
    """Full evaluation report for one role."""
    chi2, dof = evaluate.chi2_per_object(
        data.g[index].cpu().numpy(), result["g_hat"], result["w2"], data.mask[index].cpu().numpy()
    )
    return {
        "reconstruction": evaluate.reconstruction_report(
            chi2,
            dof,
            arrays["sig_logr"][index],
            arrays["tile_id"][index],
            arrays["quality"][index],
        ),
        "latent": evaluate.latent_report(result["kl"]),
        "nll": float(result["rec"].mean()),
    }


def model_spec(config: Config) -> dict[str, object]:
    return {
        "encoder": config.model.encoder,
        "n_channels": len(config.model.channels),
        "n_cond": len(config.model.conditions),
        "n_bands": N_BANDS,
        "n_narrow": len(model_data.NARROW_INDEX),
        "latent_dim": config.model.latent_dim,
        "conv_channels": list(config.model.conv_channels),
        "kernel_size": config.model.kernel_size,
        "narrow_features": config.model.narrow_features,
        "other_hidden": list(config.model.other_hidden),
        "hidden": list(config.model.hidden),
        "dropout": config.model.dropout,
    }


def checkpoint(model: VAE, config: Config, epoch: int) -> dict[str, object]:
    return {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "model": model_spec(config),
        "channels": list(config.model.channels),
        "conditions": list(config.model.conditions),
        "eps": list(config.loss.eps_per_band or (config.loss.eps,)),
        "rotation": config.data.rotation,
        "epoch": epoch,
        "config": config.to_dict(),
    }


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
    wandb_project: Annotated[
        str | None, typer.Option("--wandb-project", help="Overrides WANDB_PROJECT")
    ] = None,
    wandb_entity: Annotated[
        str | None, typer.Option("--wandb-entity", help="Overrides WANDB_ENTITY")
    ] = None,
) -> dict[str, object]:
    """Train the VAE on one rotation. ``--set dotted.key=value`` overrides the YAML."""
    config = load_config(config_path, list(overrides or []))
    if seed is not None:
        config.train.seed = seed
    if debug:
        config.train.epochs = config.train.debug_epochs
    if wandb_project:
        os.environ["WANDB_PROJECT"] = wandb_project
    if wandb_entity:
        os.environ["WANDB_ENTITY"] = wandb_entity
    with tracking(config, debug, list(overrides or [])) as tracker:
        return fit(config, config_path, list(overrides or []), debug, tracker)


def fit(
    config: Config,
    config_path: Path | None,
    overrides: list[str],
    debug: bool,
    tracker: Tracker,
) -> dict[str, object]:
    """The training run itself; ``main`` only resolves the config and tracking."""
    set_seed(config.train.seed)
    device = resolve_device(config.train.device)

    arrays = model_data.load(config.data.path)
    n_rot = len(np.unique(arrays["tile_id"]))
    if config.data.rotation >= n_rot:
        raise ValueError(f"data.rotation={config.data.rotation} but only {n_rot} rotations")
    parts = model_data.split_indices(arrays, config.data.rotation)
    rng = np.random.default_rng(config.train.seed)
    if debug:
        if config.train.debug_objects > len(parts["train"]):
            raise ValueError(
                f"debug_objects={config.train.debug_objects} exceeds the "
                f"{len(parts['train'])} training objects"
            )
        parts["train"] = np.sort(rng.choice(parts["train"], config.train.debug_objects, False))
        for role in ("val", "test"):
            keep = min(DEBUG_EVAL_OBJECTS, len(parts[role]))
            parts[role] = np.sort(rng.choice(parts[role], keep, replace=False))

    data = Tensors.build(arrays, config.model.channels, config.model.conditions, device)
    eps = floor_tensor(config, device)
    model = build_model(model_spec(config)).to(device)
    model.set_condition_scale(data.cond[parts["train"]])
    loss_fn = get_loss(config.loss.name)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )

    steps_per_epoch = math.ceil(len(parts["train"]) / config.train.batch_size)
    total_steps = steps_per_epoch * config.train.epochs
    logger.info(
        f"rotation {config.data.rotation}: {len(parts['train']):,} train / "
        f"{len(parts['val']):,} val / {len(parts['test']):,} test objects; "
        f"{config.train.epochs} epochs x {steps_per_epoch} steps on {device}"
    )

    history: list[dict[str, object]] = []
    best = {"loss": math.inf, "epoch": 0, "state": None}
    step = 0
    started = time.time()
    for epoch in range(1, config.train.epochs + 1):
        model.train()
        order = rng.permutation(parts["train"])
        running = {"loss": 0.0, "reconstruction": 0.0, "kl": 0.0}
        for start in range(0, len(order), config.train.batch_size):
            beta = beta_at(step, total_steps, config)
            terms = batch_terms(
                model,
                loss_fn,
                config,
                data.rows(order[start : start + config.train.batch_size]),
                eps,
                sample=True,
            )
            rec, kl = terms["rec"].mean(), terms["kl"].sum(-1).mean()
            loss = rec + beta * kl
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if config.train.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip)
            optimiser.step()
            step += 1
            running["loss"] += float(loss.detach())
            running["reconstruction"] += float(rec.detach())
            running["kl"] += float(kl.detach())
            if config.train.log_every and step % config.train.log_every == 0:
                logger.info(f"epoch {epoch} step {step}/{total_steps} loss {float(loss):.3f}")

        val = epoch_summary(
            run(model, loss_fn, config, data, parts["val"], eps, config.train.batch_size),
            config.loss.beta,
        )
        record = {
            "epoch": epoch,
            "beta": beta_at(step, total_steps, config),
            **{f"train_{k}": v / steps_per_epoch for k, v in running.items()},
            **{f"val_{k}": v for k, v in val.items()},
        }
        history.append(record)
        tracker.log(epoch_log(record, optimiser.param_groups[0]["lr"]), step=epoch)
        # Select only once beta has reached its final value, so the selected
        # epoch is scored by the objective it was trained for.
        warm = record["beta"] >= config.loss.beta
        if warm and val["loss"] < best["loss"]:
            best = {"loss": val["loss"], "epoch": epoch, "state": checkpoint(model, config, epoch)}
        logger.info(
            f"epoch {epoch}: train {record['train_loss']:.2f} | val {val['loss']:.2f} "
            f"(rec {val['reconstruction']:.2f}, kl {val['kl']:.2f}) beta {record['beta']:.2f}"
            + (" *" if best["epoch"] == epoch else "")
        )
        if not math.isfinite(val["loss"]):
            raise RuntimeError(f"non-finite validation loss at epoch {epoch}")

    last = checkpoint(model, config, config.train.epochs)
    if best["state"] is None:  # the run ended before beta finished warming up
        best = {"loss": history[-1]["val_loss"], "epoch": config.train.epochs, "state": last}
    model_path = config.output.model_path
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(best["state"], model_path)
    torch.save(last, last_path(model_path))

    model.load_state_dict(best["state"]["state_dict"])
    reports = {
        role: report(
            run(model, loss_fn, config, data, parts[role], eps, config.train.batch_size),
            data,
            arrays,
            parts[role],
        )
        for role in ("val", "test")
    }
    metrics = {
        "config": config.to_dict(),
        "config_source": str(config_path) if config_path else None,
        "overrides": overrides,
        "debug": debug,
        "device": str(device),
        "rotation": config.data.rotation,
        "n_objects": {role: len(parts[role]) for role in ("train", "val", "test")},
        "best_epoch": best["epoch"],
        "history": history,
        "val": reports["val"],
        "test": reports["test"],
        "elapsed_seconds": round(time.time() - started, 2),
        "model_path": str(model_path),
    }
    config.output.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    config.output.metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    tracker.summary(summary_values(metrics))
    test = reports["test"]
    logger.success(
        f"best epoch {best['epoch']}; test chi2/dof pooled "
        f"{test['reconstruction']['overall']['chi2_dof_pooled']:.3f}, "
        f"active dims {test['latent']['active_dims']}/{config.model.latent_dim}. "
        f"Wrote {model_path} and {config.output.metrics_path}"
    )
    if debug:
        logger.info("config used:\n" + dump_config(config))
    return metrics


if __name__ == "__main__":
    app()
