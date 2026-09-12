from dataclasses import dataclass, fields
from pathlib import Path

import yaml

ROOT = Path(__file__).parents[2]


@dataclass
class TransformerConfig:
    n_ctx: int
    n_vocab: int
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    d_mlp: int
    d_reps: int


@dataclass
class TrainConfig:
    lr: float
    n_warmup: int
    n_batches: int
    batch_size: int
    ema_wt: float
    wt_decay: float
    mask_wt: float
    koleo_wt: float
    grad_norm: float
    mixed_precision: str
    save_every: int
    save_path: str
    wandb_name: str
    log_every: int


def _build[T](cls: type[T], raw: dict) -> T:
    return cls(**{f.name: f.type(raw[f.name]) for f in fields(cls)})


def load_configs(
    path: Path = ROOT / "config.yaml",
) -> tuple[TransformerConfig, TrainConfig]:
    with open(path) as f:
        raw = yaml.safe_load(f)

    return _build(TransformerConfig, raw["model"]), _build(TrainConfig, raw["train"])
