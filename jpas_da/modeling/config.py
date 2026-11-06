"""Typed experiment configuration, loaded from YAML.

Everything the model and the training loop need to know lives in one nested
dataclass tree, so a run is reproducible from ``configs/vae.yaml`` plus the
CLI overrides recorded in ``reports/metrics.json``. Unknown keys fail loudly:
a typo in a YAML file must not silently fall back to a default.

Changing the channel set, the network size or the loss is a change of this
file (or of the YAML) only -- no edit to the model or the training loop.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints

import yaml

from jpas_da.config import MODELS_DIR, PROCESSED_DATA_DIR, REPORTS_DIR
from jpas_da.modeling.data import CHANNELS, CONDITIONS
from jpas_da.modeling.vae import ENCODERS


@dataclass
class DataConfig:
    path: Path = PROCESSED_DATA_DIR / "vae_inputs.npz"
    rotation: int = 0
    """Tile rotation: test = tile k, validation = tile k+1, training = the rest."""


@dataclass
class ModelConfig:
    encoder: str = "conv"
    """``conv`` (two-branch) or ``mlp`` (everything flattened; the ablation)."""

    channels: tuple[str, ...] = CHANNELS
    """Subset of ``("g", "v", "mask")`` fed to the encoder."""

    conditions: tuple[str, ...] = ("logr",)
    """Per-object conditions given to encoder and decoder; must include logr."""

    latent_dim: int = 8

    conv_channels: tuple[int, ...] = (32, 64)
    """Widths of the convolutions over the 54 narrow bands."""

    kernel_size: int = 5
    """Odd convolution width along the narrow-band axis."""

    narrow_features: int = 128
    """Width of the linear layer after flattening the narrow branch."""

    other_hidden: tuple[int, ...] = (64,)
    """MLP widths for the six off-grid bands plus the conditions."""

    hidden: tuple[int, ...] = (256, 128)
    """Encoder head widths; the decoder mirrors them in reverse."""

    dropout: float = 0.0


@dataclass
class LossConfig:
    name: str = "gaussian_nll"
    """Key of :data:`jpas_da.modeling.losses.LOSSES`."""

    beta: float = 1.0
    """KL weight after warm-up (1 = the standard ELBO)."""

    beta_warmup_fraction: float = 0.25
    """Fraction of the run over which ``beta`` ramps linearly from 0."""

    eps: float = 0.1
    """Brightness-relative floor: no band is known better than ``eps * r_bb``.

    0.1 binds on ~5% of training cells. On rotation 0 it moves the share of the
    total weight held by the top objects from 50% in 1.4% of objects (eps =
    0.01) to 50% in 5.5%, and the stars' share from 0.72 to 0.39 (see
    references/pipeline-v2-plan.md, B3)."""

    eps_per_band: tuple[float, ...] = ()
    """Optional per-band floor (60 values); overrides ``eps`` when given."""

    common_mode: bool = True
    """Add the shared error of ``r_bb``, ``(g * sig_logr)^2``, to every band."""

    student_t_dof: float = 5.0
    """Degrees of freedom when ``name == "student_t"``."""


@dataclass
class TrainConfig:
    epochs: int = 50
    batch_size: int = 512
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 0.0
    seed: int = 0
    device: str = "auto"
    """``"auto"``, ``"cpu"`` or a torch device string such as ``"cuda"``."""

    debug_objects: int = 1024
    """Training objects used with ``--debug``."""

    debug_epochs: int = 2
    """Epochs used with ``--debug``."""

    log_every: int = 0
    """Batches between progress lines; 0 disables."""


@dataclass
class OutputConfig:
    model_path: Path = MODELS_DIR / "vae.pt"
    metrics_path: Path = REPORTS_DIR / "metrics.json"


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def validate(self) -> Config:
        model, loss, train = self.model, self.loss, self.train
        if self.data.rotation < 0:
            raise ValueError("data.rotation must be >= 0")
        if model.encoder not in ENCODERS:
            raise ValueError(f"unknown encoder {model.encoder!r}; allowed: {list(ENCODERS)}")
        unknown = set(model.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels {sorted(unknown)}; allowed: {list(CHANNELS)}")
        if not model.channels:
            raise ValueError("model.channels must name at least one channel")
        if set(model.conditions) - set(CONDITIONS) or "logr" not in model.conditions:
            raise ValueError(f"model.conditions must include logr, subset of {list(CONDITIONS)}")
        if model.latent_dim < 1:
            raise ValueError("model.latent_dim must be >= 1")
        for name in ("conv_channels", "other_hidden", "hidden"):
            widths = getattr(model, name)
            if not widths or any(width < 1 for width in widths):
                raise ValueError(f"model.{name} must list at least one width >= 1")
        if model.narrow_features < 1:
            raise ValueError("model.narrow_features must be >= 1")
        if model.kernel_size < 1 or model.kernel_size % 2 == 0:
            raise ValueError("model.kernel_size must be odd and >= 1")
        if not 0.0 <= model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if loss.name not in _loss_names():
            raise ValueError(f"unknown loss {loss.name!r}; allowed: {_loss_names()}")
        if loss.beta < 0:
            raise ValueError("loss.beta must be >= 0")
        if not 0.0 <= loss.beta_warmup_fraction <= 1.0:
            raise ValueError("loss.beta_warmup_fraction must be in [0, 1]")
        if loss.eps <= 0:
            raise ValueError("loss.eps must be > 0")
        if loss.eps_per_band and (len(loss.eps_per_band) != 60 or min(loss.eps_per_band) <= 0):
            raise ValueError("loss.eps_per_band must be empty or 60 positive values")
        if loss.student_t_dof <= 0:
            raise ValueError("loss.student_t_dof must be > 0")
        if train.epochs < 1 or train.debug_epochs < 1:
            raise ValueError("epoch counts must be >= 1")
        if train.batch_size < 1 or train.debug_objects < 1:
            raise ValueError("batch_size and debug_objects must be >= 1")
        if train.learning_rate <= 0:
            raise ValueError("train.learning_rate must be > 0")
        return self

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(asdict(self))


def _loss_names() -> list[str]:
    from jpas_da.modeling.losses import LOSSES

    return sorted(LOSSES)


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _coerce(target: Any, value: Any, where: str) -> Any:
    """Coerce a YAML scalar into the annotated type of a dataclass field."""
    if target is Any or target is None:
        return value
    origin = get_origin(target)
    if is_dataclass(target):
        if not isinstance(value, dict):
            raise ValueError(f"{where}: expected a mapping, got {type(value).__name__}")
        return _build(target, value, where)
    if origin is tuple:
        args = get_args(target)
        item_type, variadic = (
            (args[0], True)
            if len(args) == 2 and args[1] is Ellipsis
            else (
                args[0] if args else Any,
                False,
            )
        )
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{where}: expected a list, got {type(value).__name__}")
        if not variadic and len(value) != len(args):
            raise ValueError(f"{where}: expected {len(args)} items, got {len(value)}")
        return tuple(_coerce(item_type, item, where) for item in value)
    if target is Path:
        return Path(value).expanduser() if isinstance(value, str) else Path(value)
    if target is bool:
        if not isinstance(value, bool):
            raise ValueError(f"{where}: expected true/false, got {value!r}")
        return value
    if target is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{where}: expected an integer, got {value!r}")
        return value
    if target is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{where}: expected a number, got {value!r}")
        return float(value)
    if target is str:
        if not isinstance(value, str):
            raise ValueError(f"{where}: expected a string, got {value!r}")
        return value
    return value


def _build(cls: type, values: dict[str, Any], where: str) -> Any:
    # ``from __future__ import annotations`` stores field types as strings, so
    # resolve them back to the classes the dataclasses were declared with.
    hints = get_type_hints(cls)
    known = {item.name: item for item in fields(cls)}
    unknown = set(values).difference(known)
    if unknown:
        raise ValueError(f"{where}: unknown keys {sorted(unknown)}; allowed {sorted(known)}")
    kwargs = {}
    for name in known:
        if name in values:
            path = f"{where}.{name}" if where else name
            kwargs[name] = _coerce(hints[name], values[name], path)
    return cls(**kwargs)


def load_config(path: Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load the YAML file at ``path`` on top of the dataclass defaults.

    ``overrides`` are ``dotted.key=value`` strings from the command line; the
    value is parsed as YAML, so ``--set model.hidden=[64,32]`` works.
    """
    values: dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"config file not found: {path}")
        loaded = yaml.safe_load(path.read_text()) or {}
        if not isinstance(loaded, dict):
            raise ValueError(f"{path}: top level must be a mapping")
        values = loaded

    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"override {override!r} is not of the form key=value")
        dotted, raw = override.split("=", 1)
        node = values
        *parents, leaf = dotted.strip().split(".")
        for part in parents:
            node = node.setdefault(part, {})
            if not isinstance(node, dict):
                raise TypeError(f"override {override!r} conflicts with {part!r}")
        node[leaf] = yaml.safe_load(raw)

    return _build(Config, values, "config").validate()


def dump_config(config: Config) -> str:
    return yaml.safe_dump(config.to_dict(), sort_keys=False)
