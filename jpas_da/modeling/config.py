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

CHANNELS = ("g", "v", "mask")
"""Encoder input channels, each one ``n_bands`` wide over the band axis."""


@dataclass
class DataConfig:
    path: Path = PROCESSED_DATA_DIR / "vae_inputs.npz"
    fold: int = 0
    """Leave-one-tile-out fold used for validation."""


@dataclass
class ModelConfig:
    channels: tuple[str, ...] = CHANNELS
    """Subset of ``("g", "v", "mask")`` fed to the encoder."""

    latent_dim: int = 6

    hidden: tuple[int, ...] = (256, 128)
    """Layer widths, one entry per convolutional encoder block."""

    kernel_size: int = 5
    """Odd convolution width along the band axis."""

    dropout: float = 0.0
    """Dropout after every hidden layer of the encoder and decoder."""

    scale_head: bool = True
    """Predict ``log r`` from the latent (brightness head)."""


@dataclass
class LossConfig:
    name: str = "gaussian_nll"
    """Key of :data:`jpas_da.modeling.losses.LOSSES`."""

    beta: float = 3.0
    """KL weight after warm-up."""

    beta_warmup_fraction: float = 0.25
    """Fraction of the run over which ``beta`` ramps linearly from 0."""

    floor_percentile: float = 5.0
    """Per-band additive error floor, as a percentile of ``v`` on train."""

    systematic_scale: float = 0.037
    """Absolute-calibration floor on ``log r``, 0.04 mag in natural log."""

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
        unknown = set(self.model.channels) - set(CHANNELS)
        if unknown:
            raise ValueError(f"unknown channels {sorted(unknown)}; allowed: {list(CHANNELS)}")
        if not self.model.channels:
            raise ValueError("model.channels must name at least one channel")
        if self.model.latent_dim < 1:
            raise ValueError("model.latent_dim must be >= 1")
        if not self.model.hidden or any(width < 1 for width in self.model.hidden):
            raise ValueError("model.hidden must list at least one width >= 1")
        if self.model.kernel_size < 1 or self.model.kernel_size % 2 == 0:
            raise ValueError("model.kernel_size must be odd and >= 1")
        if not 0.0 <= self.model.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")
        if self.loss.name not in _loss_names():
            raise ValueError(f"unknown loss {self.loss.name!r}; allowed: {_loss_names()}")
        if self.loss.beta < 0:
            raise ValueError("loss.beta must be >= 0")
        if not 0.0 <= self.loss.beta_warmup_fraction <= 1.0:
            raise ValueError("loss.beta_warmup_fraction must be in [0, 1]")
        if not 0.0 <= self.loss.floor_percentile <= 100.0:
            raise ValueError("loss.floor_percentile must be in [0, 100]")
        if self.loss.systematic_scale <= 0:
            raise ValueError("loss.systematic_scale must be > 0")
        if self.loss.student_t_dof <= 0:
            raise ValueError("loss.student_t_dof must be > 0")
        if self.train.epochs < 1 or self.train.debug_epochs < 1:
            raise ValueError("epoch counts must be >= 1")
        if self.train.batch_size < 1 or self.train.debug_objects < 1:
            raise ValueError("batch_size and debug_objects must be >= 1")
        if self.train.learning_rate <= 0:
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
